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
PERSON_CONFIDENCE_THRESHOLD = 0.30  # 人员检测置信度阈值（稍低以提高召回率）
PERSON_MIN_BBOX_RATIO = 0.005  # 人员边界框最小占比（过滤极小误检）

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


class PersonDetectResponse(BaseModel):
    success: bool
    has_person: bool      # 是否检测到人员
    person_count: int      # 检测到的人员数量
    detections: list[Detection]
    alert_level: str       # 告警级别: none / warning / critical
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
# 人员检测接口（周界围挡入侵检测）
# ---------------------------------------------------------------------------
@app.post("/detect/person", response_model=PersonDetectResponse)
async def detect_person(file: UploadFile = File(...)):
    """
    专门的人员检测接口，用于周界围挡等场景的人员入侵识别。
    只检测 Person 类别，返回是否有人及告警级别。
    """
    if file.content_type and file.content_type not in (
        "image/jpeg", "image/png", "image/bmp", "image/webp",
    ):
        raise HTTPException(400, f"不支持的图片格式: {file.content_type}")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"无法解析图片: {str(e)}")

    img_w, img_h = image.size
    img_area = img_w * img_h

    # 推理
    results = model(image, conf=PERSON_CONFIDENCE_THRESHOLD, verbose=False)
    result = results[0]

    detections: list[Detection] = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = model.names[cls_id]

            # 只关注 Person 类别
            if cls_name != "Person":
                continue

            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxyn[0].tolist()

            # 过滤极小目标（bbox 面积占比过小，通常是误检）
            bbox_w = (x2 - x1) * img_w
            bbox_h = (y2 - y1) * img_h
            bbox_area_ratio = (bbox_w * bbox_h) / img_area
            if bbox_area_ratio < PERSON_MIN_BBOX_RATIO:
                continue

            detections.append(Detection(
                class_name=cls_name,
                label="人员",
                confidence=round(conf, 4),
                bbox=[round(v, 4) for v in [x1, y1, x2, y2]],
            ))

    detections.sort(key=lambda d: d.confidence, reverse=True)

    person_count = len(detections)
    has_person = person_count > 0

    # 告警级别
    if person_count == 0:
        alert_level = "none"
        message = "未检测到人员"
    elif person_count == 1:
        alert_level = "warning"
        message = f"检测到 1 名人员 (置信度: {detections[0].confidence:.2f})"
    else:
        alert_level = "critical"
        confs = ", ".join(f"{d.confidence:.2f}" for d in detections)
        message = f"检测到 {person_count} 名人员 (置信度: {confs})"

    return PersonDetectResponse(
        success=True,
        has_person=has_person,
        person_count=person_count,
        detections=detections,
        alert_level=alert_level,
        message=message,
    )


# ---------------------------------------------------------------------------
# Web 测试页面
# ---------------------------------------------------------------------------
@app.get("/demo", response_class=HTMLResponse)
def demo_page():
    return DEMO_HTML


@app.get("/demo/person", response_class=HTMLResponse)
def demo_person_page():
    return DEMO_PERSON_HTML


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


DEMO_PERSON_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>人员入侵检测 Demo</title>
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
    margin-bottom: 4px;
    background: linear-gradient(135deg, #f87171, #fb923c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 14px;
    margin-bottom: 24px;
  }
  .badge-row {
    display: flex;
    gap: 10px;
    justify-content: center;
    margin-bottom: 24px;
  }
  .badge {
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
  }
  .badge-safe   { background: rgba(74,222,128,0.15); color: #4ade80; border: 1px solid rgba(74,222,128,0.3); }
  .badge-warn   { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
  .badge-danger { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }

  .dropzone {
    border: 2px dashed #475569;
    border-radius: 12px;
    padding: 40px 20px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    margin-bottom: 20px;
  }
  .dropzone:hover, .dropzone.dragover {
    border-color: #f87171;
    background: rgba(248,113,113,0.05);
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
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    text-align: center;
  }
  .result.show { display: block; }
  .result.none {
    background: rgba(74,222,128,0.1);
    border: 1px solid rgba(74,222,128,0.3);
  }
  .result.warning {
    background: rgba(251,191,36,0.1);
    border: 1px solid rgba(251,191,36,0.3);
  }
  .result.critical {
    background: rgba(248,113,113,0.1);
    border: 1px solid rgba(248,113,113,0.3);
  }
  .result.error {
    background: rgba(148,163,184,0.1);
    border: 1px solid rgba(148,163,184,0.2);
  }
  .result-icon { font-size: 48px; margin-bottom: 8px; }
  .result-title { font-size: 20px; font-weight: 700; margin-bottom: 4px; }
  .result-none .result-title { color: #4ade80; }
  .result-warn .result-title { color: #fbbf24; }
  .result-danger .result-title { color: #f87171; }
  .result-detail { font-size: 14px; color: #94a3b8; margin-top: 8px; }
  .detection-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    margin: 4px;
    font-size: 13px;
  }
  .detection-dot {
    width: 8px; height: 8px;
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
    background: linear-gradient(135deg, #f87171, #fb923c);
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
    color: #fb923c;
    font-size: 14px;
  }
  .loading.show { display: block; }
  .spinner {
    display: inline-block;
    width: 18px; height: 18px;
    border: 2px solid rgba(251,146,60,0.3);
    border-top-color: #fb923c;
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
  <h1>🚨 人员入侵检测</h1>
  <p class="subtitle">上传周界围挡图片，AI 自动判断是否有人入侵</p>

  <div class="badge-row">
    <span class="badge badge-safe">🟢 安全</span>
    <span class="badge badge-warn">🟡 警告</span>
    <span class="badge badge-danger">🔴 危险</span>
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

  <div class="footer">YOLOv8s · Open Images V7 · 人员检测专版</div>
</div>

<script>
const PERSON_COLOR = '#fb923c';
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
dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
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
    result.classList.remove('show', 'none', 'warning', 'critical', 'error');
    result.innerHTML = '';
    clearCanvas();
  };
  reader.readAsDataURL(file);
}

btnDetect.addEventListener('click', async () => {
  if (!selectedFile) return;
  btnDetect.disabled = true;
  loading.classList.add('show');
  result.classList.remove('show', 'none', 'warning', 'critical', 'error');
  clearCanvas();

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const resp = await fetch('/detect/person', { method: 'POST', body: formData });
    const data = await resp.json();

    if (!data.success) {
      result.className = 'result show error';
      result.innerHTML = '<div class="result-icon">❌</div><div class="result-title">检测失败</div>';
    } else if (!data.has_person) {
      result.className = 'result show none';
      result.innerHTML = '<div class="result-icon">✅</div><div class="result-title" style="color:#4ade80">安全 · 未检测到人员</div><div class="result-detail">周界区域无人员入侵</div>';
    } else {
      const levelClass = data.alert_level === 'critical' ? 'critical' : 'warning';
      const icon = data.alert_level === 'critical' ? '🚨' : '⚠️';
      const titleColor = data.alert_level === 'critical' ? '#f87171' : '#fbbf24';
      const titleText = data.alert_level === 'critical' ? '危险 · 多人入侵' : '警告 · 检测到人员';

      let html = '<div class="result-icon">' + icon + '</div>';
      html += '<div class="result-title" style="color:' + titleColor + '">' + titleText + '</div>';
      html += '<div class="result-detail">检测到 ' + data.person_count + ' 名人员</div>';
      if (data.detections.length > 0) {
        html += '<div class="result-detail">';
        data.detections.forEach(d => {
          html += '<span class="detection-item">' +
            '<span class="detection-dot" style="background:' + PERSON_COLOR + '"></span>' +
            '人员 ' + (d.confidence * 100).toFixed(1) + '%</span>';
        });
        html += '</div>';
      }
      result.className = 'result show ' + levelClass;
      result.innerHTML = html;
      drawBBoxes(data.detections);
    }
  } catch (err) {
    result.className = 'result show error';
    result.innerHTML = '<div class="result-icon">❌</div><div class="result-title">请求失败</div><div class="result-detail">' + err.message + '</div>';
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
  result.classList.remove('show', 'none', 'warning', 'critical', 'error');
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
    const px = x1 * canvas.width;
    const py = y1 * canvas.height;
    const pw = (x2 - x1) * canvas.width;
    const ph = (y2 - y1) * canvas.height;

    ctx.strokeStyle = PERSON_COLOR;
    ctx.lineWidth = 3;
    ctx.strokeRect(px, py, pw, ph);

    ctx.fillStyle = PERSON_COLOR;
    ctx.font = 'bold 14px -apple-system, sans-serif';
    const label = '人员 ' + (d.confidence * 100).toFixed(0) + '%';
    const tm = ctx.measureText(label);
    const lw = tm.width + 10;
    const lh = 22;
    ctx.fillRect(px, py - lh, lw, lh);
    ctx.fillStyle = '#0f172a';
    ctx.fillText(label, px + 5, py - 6);
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
