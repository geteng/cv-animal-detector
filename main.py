"""
CV Animal Detector - 动物检测 API 服务
识别：人类、爬行动物（蛇、蜥蜴）、小型兽类（猫、狗、野猪）
基于 YOLOv8 Open Images V7 预训练模型
"""

import io
import logging
from typing import Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
MODEL_NAME = "yolov8s-oiv7.pt"  # 首次运行会自动下载 (~24MB)

# Open Images V7 中对应类别的标签名
TARGET_CLASSES = {
    "Person":     "人类",
    "Snake":      "蛇",
    "Lizard":     "蜥蜴",
    "Cat":        "猫",
    "Dog":        "狗",
    "Wild boar":  "野猪",
}

CONFIDENCE_THRESHOLD = 0.35  # 置信度阈值

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cv-detector")

# ---------------------------------------------------------------------------
# 加载模型（全局单例）
# ---------------------------------------------------------------------------
logger.info(f"正在加载模型 {MODEL_NAME} ...")
model = YOLO(MODEL_NAME)
logger.info("模型加载完成")

# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CV Animal Detector",
    description="检测图片中的人类、爬行动物（蛇/蜥蜴）和小型兽类（猫/狗/野猪）",
    version="1.0.0",
)


class Detection(BaseModel):
    class_name: str       # 英文类别名
    label: str            # 中文标签
    confidence: float     # 置信度 0-1
    bbox: list[float]     # [x1, y1, x2, y2] 归一化坐标 (0-1)


class DetectResponse(BaseModel):
    success: bool
    has_animal: bool
    detections: list[Detection]
    count: int
    message: str


@app.get("/")
def root():
    return {"service": "CV Animal Detector", "version": "1.0.0", "status": "running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_NAME}


@app.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...)):
    """
    上传图片，返回检测结果。
    支持的格式：JPEG, PNG, BMP, WEBP
    """
    # 校验文件类型
    if file.content_type and file.content_type not in (
        "image/jpeg", "image/png", "image/bmp", "image/webp",
    ):
        raise HTTPException(400, f"不支持的图片格式: {file.content_type}")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"无法解析图片: {str(e)}")

    # 推理
    results = model(image, conf=CONFIDENCE_THRESHOLD, verbose=False)
    result = results[0]

    detections: list[Detection] = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            cls_name = model.names[cls_id]

            if cls_name not in TARGET_CLASSES:
                continue

            # 归一化坐标
            x1, y1, x2, y2 = box.xyxyn[0].tolist()
            detections.append(Detection(
                class_name=cls_name,
                label=TARGET_CLASSES[cls_name],
                confidence=round(conf, 4),
                bbox=[round(v, 4) for v in [x1, y1, x2, y2]],
            ))

    # 按置信度降序排列
    detections.sort(key=lambda d: d.confidence, reverse=True)

    has_animal = len(detections) > 0
    if has_animal:
        labels = ", ".join(f"{d.label}({d.confidence:.2f})" for d in detections)
        message = f"检测到目标: {labels}"
    else:
        message = "未检测到目标动物或人类"

    return DetectResponse(
        success=True,
        has_animal=has_animal,
        detections=detections,
        count=len(detections),
        message=message,
    )


@app.post("/detect/batch", response_model=list[DetectResponse])
async def detect_batch(files: list[UploadFile] = File(...)):
    """批量检测，最多 10 张"""
    if len(files) > 10:
        raise HTTPException(400, "单次最多上传 10 张图片")

    results_list = []
    for file in files:
        try:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            results_list.append(DetectResponse(
                success=False, has_animal=False, detections=[], count=0,
                message=f"无法解析图片: {file.filename}",
            ))
            continue

        results = model(image, conf=CONFIDENCE_THRESHOLD, verbose=False)
        result = results[0]

        dets = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_name = model.names[int(box.cls[0].item())]
                if cls_name not in TARGET_CLASSES:
                    continue
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxyn[0].tolist()
                dets.append(Detection(
                    class_name=cls_name, label=TARGET_CLASSES[cls_name],
                    confidence=round(conf, 4),
                    bbox=[round(v, 4) for v in [x1, y1, x2, y2]],
                ))

        dets.sort(key=lambda d: d.confidence, reverse=True)
        has = len(dets) > 0
        results_list.append(DetectResponse(
            success=True, has_animal=has, detections=dets, count=len(dets),
            message=f"检测到: {', '.join(d.label for d in dets)}" if has else "未检测到目标",
        ))

    return results_list


# ---------------------------------------------------------------------------
# Web 测试页面
# ---------------------------------------------------------------------------
@app.get("/demo", response_class=HTMLResponse)
def demo_page():
    return DEMO_HTML


DEMO_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>动物检测 Demo</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
  }
  .container {
    width: 100%;
    max-width: 720px;
    background: #1e293b;
    border-radius: 16px;
    padding: 32px;
    box-shadow: 0 25px 50px rgba(0,0,0,0.4);
  }
  h1 {
    font-size: 24px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 8px;
    background: linear-gradient(135deg, #38bdf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 14px;
    margin-bottom: 28px;
  }
  .tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin-bottom: 24px;
  }
  .tag {
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 500;
  }
  .tag-person  { background: rgba(56,189,248,0.15); color: #38bdf8; }
  .tag-snake  { background: rgba(74,222,128,0.15); color: #4ade80; }
  .tag-lizard { background: rgba(251,191,36,0.15); color: #fbbf24; }
  .tag-cat    { background: rgba(244,114,182,0.15); color: #f472b6; }
  .tag-dog    { background: rgba(167,139,250,0.15); color: #a78bfa; }
  .tag-boar   { background: rgba(251,146,60,0.15); color: #fb923c; }

  .dropzone {
    border: 2px dashed #475569;
    border-radius: 12px;
    padding: 40px 20px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    margin-bottom: 20px;
    position: relative;
  }
  .dropzone:hover, .dropzone.dragover {
    border-color: #38bdf8;
    background: rgba(56,189,248,0.05);
  }
  .dropzone-icon { font-size: 40px; margin-bottom: 12px; }
  .dropzone-text { color: #94a3b8; font-size: 15px; }
  .dropzone-hint { color: #64748b; font-size: 13px; margin-top: 6px; }
  #fileInput { display: none; }

  .preview-area {
    display: none;
    position: relative;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 20px;
    background: #0f172a;
  }
  .preview-area.show { display: block; }
  .preview-area img {
    width: 100%;
    display: block;
    max-height: 480px;
    object-fit: contain;
  }
  .preview-area canvas {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
  }

  .result {
    display: none;
    padding: 16px;
    border-radius: 10px;
    margin-bottom: 20px;
    font-size: 14px;
  }
  .result.show { display: block; }
  .result.found {
    background: rgba(74,222,128,0.1);
    border: 1px solid rgba(74,222,128,0.25);
    color: #4ade80;
  }
  .result.empty {
    background: rgba(148,163,184,0.1);
    border: 1px solid rgba(148,163,184,0.2);
    color: #94a3b8;
  }
  .result.error {
    background: rgba(248,113,113,0.1);
    border: 1px solid rgba(248,113,113,0.25);
    color: #f87171;
  }
  .result-title { font-weight: 600; margin-bottom: 8px; }
  .detection-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 0;
  }
  .detection-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .btn-row {
    display: flex;
    gap: 12px;
  }
  .btn {
    flex: 1;
    padding: 12px;
    border: none;
    border-radius: 10px;
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
  }
  .btn-detect {
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    color: #fff;
  }
  .btn-detect:hover { opacity: 0.9; transform: translateY(-1px); }
  .btn-detect:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
  .btn-reset {
    background: #334155;
    color: #cbd5e1;
  }
  .btn-reset:hover { background: #475569; }

  .loading {
    display: none;
    text-align: center;
    padding: 12px;
    color: #38bdf8;
    font-size: 14px;
  }
  .loading.show { display: block; }
  .spinner {
    display: inline-block;
    width: 18px; height: 18px;
    border: 2px solid rgba(56,189,248,0.3);
    border-top-color: #38bdf8;
    border-radius: 50%;
    animation: spin 0.6s linear infinite;
    vertical-align: middle;
    margin-right: 8px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .footer {
    text-align: center;
    color: #475569;
    font-size: 12px;
    margin-top: 20px;
  }
</style>
</head>
<body>
<div class="container">
  <h1>🐾 动物检测 Demo</h1>
  <p class="subtitle">上传图片，AI 自动识别目标</p>

  <div class="tags">
    <span class="tag tag-person">🧑 人类</span>
    <span class="tag tag-snake">🐍 蛇</span>
    <span class="tag tag-lizard">🦎 蜥蜴</span>
    <span class="tag tag-cat">🐱 猫</span>
    <span class="tag tag-dog">🐶 狗</span>
    <span class="tag tag-boar">🐗 野猪</span>
  </div>

  <div class="dropzone" id="dropzone">
    <div class="dropzone-icon">📁</div>
    <div class="dropzone-text">点击选择图片或拖拽到此处</div>
    <div class="dropzone-hint">支持 JPG / PNG / BMP / WEBP</div>
    <input type="file" id="fileInput" accept="image/jpeg,image/png,image/bmp,image/webp">
  </div>

  <div class="preview-area" id="previewArea">
    <img id="previewImg" alt="预览">
    <canvas id="bboxCanvas"></canvas>
  </div>

  <div class="loading" id="loading">
    <span class="spinner"></span>正在检测中...
  </div>

  <div class="result" id="result"></div>

  <div class="btn-row">
    <button class="btn btn-detect" id="btnDetect" disabled>🔍 开始检测</button>
    <button class="btn btn-reset" id="btnReset">🔄 重新选择</button>
  </div>

  <div class="footer">YOLOv8s · Open Images V7</div>
</div>

<script>
const COLORS = {
  'Person':    '#38bdf8',
  'Snake':     '#4ade80',
  'Lizard':    '#fbbf24',
  'Cat':       '#f472b6',
  'Dog':       '#a78bfa',
  'Wild boar': '#fb923c',
};

let selectedFile = null;

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const previewArea = document.getElementById('previewArea');
const previewImg = document.getElementById('previewImg');
const bboxCanvas = document.getElementById('bboxCanvas');
const loading = document.getElementById('loading');
const result = document.getElementById('result');
const btnDetect = document.getElementById('btnDetect');
const btnReset = document.getElementById('btnReset');

dropzone.addEventListener('click', () => fileInput.click());

dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('dragover');
});
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  const files = e.dataTransfer.files;
  if (files.length > 0) handleFile(files[0]);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  if (!file.type.match(/image\/(jpeg|png|bmp|webp)/)) {
    alert('不支持的格式，请选择 JPG/PNG/BMP/WEBP 图片');
    return;
  }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewArea.classList.add('show');
    dropzone.style.display = 'none';
    btnDetect.disabled = false;
    result.classList.remove('show', 'found', 'empty', 'error');
    result.innerHTML = '';
    clearCanvas();
  };
  reader.readAsDataURL(file);
}

btnDetect.addEventListener('click', async () => {
  if (!selectedFile) return;

  btnDetect.disabled = true;
  loading.classList.add('show');
  result.classList.remove('show', 'found', 'empty', 'error');
  clearCanvas();

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const resp = await fetch('/detect', { method: 'POST', body: formData });
    const data = await resp.json();

    if (data.has_animal) {
      result.className = 'result show found';
      let html = '<div class="result-title">✅ ' + data.message + '</div>';
      data.detections.forEach(d => {
        const color = COLORS[d.class_name] || '#fff';
        html += '<div class="detection-item">' +
          '<span class="detection-dot" style="background:' + color + '"></span>' +
          '<span>' + d.label + ' (' + d.class_name + ') — 置信度 ' +
          (d.confidence * 100).toFixed(1) + '%</span>' +
          '</div>';
      });
      result.innerHTML = html;
      drawBBoxes(data.detections);
    } else {
      result.className = 'result show empty';
      result.innerHTML = '<div class="result-title">🔍 未检测到目标动物或人类</div>';
    }
  } catch (err) {
    result.className = 'result show error';
    result.innerHTML = '<div class="result-title">❌ 检测失败: ' + err.message + '</div>';
  } finally {
    loading.classList.remove('show');
    btnDetect.disabled = false;
  }
});

btnReset.addEventListener('click', () => {
  selectedFile = null;
  fileInput.value = '';
  previewArea.classList.remove('show');
  previewImg.src = '';
  dropzone.style.display = '';
  btnDetect.disabled = true;
  result.classList.remove('show', 'found', 'empty', 'error');
  result.innerHTML = '';
  clearCanvas();
});

function drawBBoxes(detections) {
  const img = previewImg;
  const canvas = bboxCanvas;
  canvas.width = img.offsetWidth;
  canvas.height = img.offsetHeight;
  const ctx = canvas.getContext('2d');

  detections.forEach(d => {
    const [x1, y1, x2, y2] = d.bbox;
    const color = COLORS[d.class_name] || '#fff';
    const px = x1 * canvas.width;
    const py = y1 * canvas.height;
    const pw = (x2 - x1) * canvas.width;
    const ph = (y2 - y1) * canvas.height;

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(px, py, pw, ph);

    ctx.fillStyle = color;
    ctx.font = 'bold 13px -apple-system, sans-serif';
    const label = d.label + ' ' + (d.confidence * 100).toFixed(0) + '%';
    const tm = ctx.measureText(label);
    const lw = tm.width + 8;
    const lh = 20;
    ctx.fillRect(px, py - lh, lw, lh);
    ctx.fillStyle = '#0f172a';
    ctx.fillText(label, px + 4, py - 5);
  });
}

function clearCanvas() {
  const ctx = bboxCanvas.getContext('2d');
  ctx.clearRect(0, 0, bboxCanvas.width, bboxCanvas.height);
}
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# 启动入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
