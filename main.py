"""
CV Detector - 多场景视觉检测 API
基于 YOLOv8s Open Images V7 预训练模型
"""

import io
import logging
from typing import Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
MODEL_NAME = "yolov8s-oiv7.pt"

TARGET_CLASSES = {
    "Person": "人类", "Snake": "蛇", "Lizard": "蜥蜴",
    "Cat": "猫", "Dog": "狗", "Wild boar": "野猪",
}

DEFAULT_CONF = 0.35
PERSON_CONF = 0.30
MIN_BBOX_RATIO = 0.005

# ---------------------------------------------------------------------------
# 日志 & 模型
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cv-detector")
logger.info(f"Loading model {MODEL_NAME} ...")
model = YOLO(MODEL_NAME)
logger.info("Model loaded")

app = FastAPI(title="CV Detector", version="2.0.0")

# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------
class Detection(BaseModel):
    class_name: str; label: str; confidence: float; bbox: list[float]

class DetectResponse(BaseModel):
    success: bool; has_animal: bool; detections: list[Detection]; count: int; message: str

class PersonDetectResponse(BaseModel):
    success: bool; has_person: bool; person_count: int; detections: list[Detection]; alert_level: str; message: str

class HelmetDetectResponse(BaseModel):
    success: bool; total_persons: int; persons_with_helmet: int; persons_without_helmet: int
    detections: list[Detection]; alert_level: str; message: str

class ClothingDetectResponse(BaseModel):
    success: bool; total_persons: int; persons_with_clothing: int; persons_without_clothing: int
    detections: list[Detection]; alert_level: str; message: str

class SleepDetectResponse(BaseModel):
    success: bool; has_sleeping_person: bool; sleeping_count: int; detections: list[Detection]; alert_level: str; message: str

class FightDetectResponse(BaseModel):
    success: bool; has_fight: bool; person_count: int; fight_groups: int; detections: list[Detection]; alert_level: str; message: str

class VehicleDetectResponse(BaseModel):
    success: bool; has_vehicle: bool; vehicle_count: int; detections: list[Detection]; alert_level: str; message: str

class GenericDetectResponse(BaseModel):
    success: bool; has_target: bool; target_count: int; detections: list[Detection]; alert_level: str; message: str

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def allowed_image(file: UploadFile):
    if file.content_type and file.content_type not in ("image/jpeg","image/png","image/bmp","image/webp"):
        raise HTTPException(400, f"Unsupported format: {file.content_type}")

async def read_image(file: UploadFile):
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        return img, img.size[0], img.size[1]
    except Exception as e:
        raise HTTPException(400, f"Cannot parse image: {e}")

def run_inference(image, conf=DEFAULT_CONF):
    return model(image, conf=conf, verbose=False)[0]

def extract_detections(result, class_filter=None, min_bbox_ratio=0.0, img_w=1, img_h=1):
    dets = []
    if result.boxes is None:
        return dets
    area = img_w * img_h
    for box in result.boxes:
        cls_name = model.names[int(box.cls[0].item())]
        if class_filter and cls_name not in class_filter:
            continue
        conf = float(box.conf[0].item())
        x1,y1,x2,y2 = box.xyxyn[0].tolist()
        if min_bbox_ratio > 0:
            bw,bh = (x2-x1)*img_w, (y2-y1)*img_h
            if (bw*bh)/area < min_bbox_ratio:
                continue
        dets.append(Detection(class_name=cls_name, label=cls_name, confidence=round(conf,4), bbox=[round(v,4) for v in [x1,y1,x2,y2]]))
    dets.sort(key=lambda d: d.confidence, reverse=True)
    return dets

def bbox_iou(b1,b2):
    x1,y1 = max(b1[0],b2[0]), max(b1[1],b2[1])
    x2,y2 = min(b1[2],b2[2]), min(b1[3],b2[3])
    inter = max(0,x2-x1)*max(0,y2-y1)
    a1,a2 = (b1[2]-b1[0])*(b1[3]-b1[1]), (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter+1e-6)

def bbox_center(b):
    return ((b[0]+b[2])/2, (b[1]+b[3])/2)

def bbox_distance(b1,b2):
    c1,c2 = bbox_center(b1), bbox_center(b2)
    return ((c1[0]-c2[0])**2+(c1[1]-c2[1])**2)**0.5

def bbox_aspect_ratio(b):
    w,h = b[2]-b[0], b[3]-b[1]
    return w/h if h>0 else 0

# ---------------------------------------------------------------------------
# 基础路由
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"service":"CV Detector","version":"2.0.0","status":"running"}

@app.get("/health")
def health():
    return {"status":"healthy","model":MODEL_NAME}

# ===================================================================
# 1. 动物+人类检测
# ===================================================================
@app.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...)):
    allowed_image(file)
    image, img_w, img_h = await read_image(file)
    result = run_inference(image)
    dets = extract_detections(result, class_filter=set(TARGET_CLASSES.keys()))
    for d in dets: d.label = TARGET_CLASSES.get(d.class_name, d.class_name)
    has = len(dets) > 0
    return DetectResponse(success=True, has_animal=has, detections=dets, count=len(dets),
        message=f"检测到目标: {', '.join(f'{d.label}({d.confidence:.2f})' for d in dets)}" if has else "未检测到目标动物或人类")

# ===================================================================
# 2. 批量检测
# ===================================================================
@app.post("/detect/batch", response_model=list[DetectResponse])
async def detect_batch(files: list[UploadFile] = File(...)):
    if len(files) > 10: raise HTTPException(400, "最多10张")
    out = []
    for file in files:
        try: image, _, _ = await read_image(file)
        except: out.append(DetectResponse(success=False, has_animal=False, detections=[], count=0, message=f"无法解析: {file.filename}")); continue
        result = run_inference(image)
        dets = extract_detections(result, class_filter=set(TARGET_CLASSES.keys()))
        for d in dets: d.label = TARGET_CLASSES.get(d.class_name, d.class_name)
        has = len(dets) > 0
        out.append(DetectResponse(success=True, has_animal=has, detections=dets, count=len(dets),
            message=f"检测到: {', '.join(d.label for d in dets)}" if has else "未检测到目标"))
    return out

# ===================================================================
# 3. 人员入侵检测
# ===================================================================
@app.post("/detect/person", response_model=PersonDetectResponse)
async def detect_person(file: UploadFile = File(...)):
    allowed_image(file)
    image, img_w, img_h = await read_image(file)
    result = run_inference(image, conf=PERSON_CONF)
    dets = extract_detections(result, class_filter={"Person"}, min_bbox_ratio=MIN_BBOX_RATIO, img_w=img_w, img_h=img_h)
    for d in dets: d.label = "人员"
    n = len(dets)
    if n == 0: return PersonDetectResponse(success=True, has_person=False, person_count=0, detections=[], alert_level="none", message="未检测到人员")
    elif n == 1: return PersonDetectResponse(success=True, has_person=True, person_count=1, detections=dets, alert_level="warning", message=f"检测到 1 名人员")
    else: return PersonDetectResponse(success=True, has_person=True, person_count=n, detections=dets, alert_level="critical", message=f"检测到 {n} 名人员")

# ===================================================================
# 4. 未佩戴安全帽检测（优化版）
# ===================================================================
@app.post("/detect/helmet", response_model=HelmetDetectResponse)
async def detect_helmet(file: UploadFile = File(...)):
    allowed_image(file)
    image, img_w, img_h = await read_image(file)
    result = run_inference(image, conf=0.2)
    persons = extract_detections(result, class_filter={"Person"}, min_bbox_ratio=MIN_BBOX_RATIO, img_w=img_w, img_h=img_h)
    helmets = extract_detections(result, class_filter={"Helmet","Bicycle helmet","Football helmet"})
    heads = extract_detections(result, class_filter={"Human head"})
    persons_with, persons_without = 0, 0
    all_dets = []
    for p in persons:
        pb = p.bbox
        ph = pb[3] - pb[1]
        head_top = pb[1]; head_bottom = pb[1] + ph * 0.3
        head_left = pb[0] + ph * 0.1; head_right = pb[2] - ph * 0.1
        has = False
        for h in helmets:
            hc = bbox_center(h.bbox)
            if head_left <= hc[0] <= head_right and head_top <= hc[1] <= head_bottom:
                has = True; break
        if not has:
            for hd in heads:
                if bbox_iou(pb, hd.bbox) > 0.05:
                    for h in helmets:
                        if bbox_distance(hd.bbox, h.bbox) < 0.15:
                            has = True; break
                    if has: break
        if not has:
            for h in helmets:
                if bbox_iou(pb, h.bbox) > 0.08:
                    hc = bbox_center(h.bbox)
                    if hc[1] < pb[1] + ph * 0.35:
                        has = True; break
        p.label = "已佩戴安全帽" if has else "未佩戴安全帽"
        if has: persons_with += 1
        else: persons_without += 1
        all_dets.append(p)
    total = len(persons)
    if total == 0: return HelmetDetectResponse(success=True, total_persons=0, persons_with_helmet=0, persons_without_helmet=0, detections=[], alert_level="none", message="未检测到人员")
    elif persons_without == 0: return HelmetDetectResponse(success=True, total_persons=total, persons_with_helmet=persons_with, persons_without_helmet=0, detections=all_dets, alert_level="none", message=f"检测到 {total} 名人员，全部已佩戴安全帽")
    else: return HelmetDetectResponse(success=True, total_persons=total, persons_with_helmet=persons_with, persons_without_helmet=persons_without, detections=all_dets, alert_level="critical" if persons_without>=2 else "warning", message=f"检测到 {total} 名人员，其中 {persons_without} 名未佩戴安全帽")

# ===================================================================
# 5. 未穿戴反光衣检测（优化版 - HSV 色彩空间）
# ===================================================================
@app.post("/detect/reflective-vest", response_model=ClothingDetectResponse)
async def detect_reflective_vest(file: UploadFile = File(...)):
    allowed_image(file)
    image, img_w, img_h = await read_image(file)
    img_np = np.array(image)
    # HSV 转换用于更准确的色彩判断
    try:
        import cv2
        img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    except ImportError:
        img_hsv = None
    result = run_inference(image, conf=0.2)
    persons = extract_detections(result, class_filter={"Person"}, min_bbox_ratio=MIN_BBOX_RATIO, img_w=img_w, img_h=img_h)
    clothing = extract_detections(result, class_filter={"Clothing","Jacket","Shirt","Coat","Sports uniform"})
    persons_with, persons_without = 0, 0
    all_dets = []
    for p in persons:
        pb = p.bbox
        px1,py1 = int(pb[0]*img_w), int(pb[1]*img_h)
        px2,py2 = int(pb[2]*img_w), int(pb[3]*img_h)
        my1 = int(py1+(py2-py1)*0.2); my2 = int(py1+(py2-py1)*0.55)
        is_bright = False
        if my2>my1 and px2>px1:
            if img_hsv is not None:
                torso_hsv = img_hsv[my1:my2, px1:px2, :]
                if torso_hsv.size > 0:
                    # 荧光黄/绿: H~25-45, S>100, V>150
                    # 荧光橙: H~8-20, S>120, V>150
                    h, s, v = torso_hsv[:,:,0], torso_hsv[:,:,1], torso_hsv[:,:,2]
                    mask_yellow = (h>=20)&(h<=45)&(s>80)&(v>140)
                    mask_orange = (h>=5)&(h<=20)&(s>100)&(v>140)
                    bright_ratio = (mask_yellow.sum() + mask_orange.sum()) / mask_yellow.size
                    is_bright = bright_ratio > 0.15
            else:
                torso = img_np[my1:my2, px1:px2, :]
                if torso.size > 0:
                    avg = torso.mean(axis=(0,1))
                    is_bright = (avg[0]>140 and avg[1]>110) or (avg[0]>170 and avg[1]>90 and avg[2]<130)
        has_cloth = any(bbox_iou(pb, c.bbox)>0.25 for c in clothing)
        has_vest = is_bright or has_cloth
        p.label = "已穿戴反光衣" if has_vest else "未穿戴反光衣"
        if has_vest: persons_with += 1
        else: persons_without += 1
        all_dets.append(p)
    total = len(persons)
    if total == 0: return ClothingDetectResponse(success=True, total_persons=0, persons_with_clothing=0, persons_without_clothing=0, detections=[], alert_level="none", message="未检测到人员")
    elif persons_without == 0: return ClothingDetectResponse(success=True, total_persons=total, persons_with_clothing=persons_with, persons_without_clothing=0, detections=all_dets, alert_level="none", message=f"检测到 {total} 名人员，全部已穿戴反光衣")
    else: return ClothingDetectResponse(success=True, total_persons=total, persons_with_clothing=persons_with, persons_without_clothing=persons_without, detections=all_dets, alert_level="critical" if persons_without>=2 else "warning", message=f"检测到 {total} 名人员，其中 {persons_without} 名未穿戴反光衣")

# ===================================================================
# 6. 未穿工服检测（优化版 - 降低阈值 + 扩展衣物类别）
# ===================================================================
@app.post("/detect/uniform", response_model=ClothingDetectResponse)
async def detect_uniform(file: UploadFile = File(...)):
    allowed_image(file)
    image, img_w, img_h = await read_image(file)
    result = run_inference(image, conf=0.2)
    persons = extract_detections(result, class_filter={"Person"}, min_bbox_ratio=MIN_BBOX_RATIO, img_w=img_w, img_h=img_h)
    # 扩展衣物类别，覆盖更多工服类型
    clothing = extract_detections(result, class_filter={
        "Clothing","Jacket","Shirt","Coat","Sports uniform","Suit","Trousers",
        "Jeans","Dress","Shorts","Footwear","Glove","Belt"
    })
    persons_with, persons_without = 0, 0
    all_dets = []
    for p in persons:
        pb = p.bbox
        # 检查是否有衣物在人员上半身区域（0%-60%）
        upper_body = (pb[0], pb[1], pb[2], pb[1] + (pb[3]-pb[1])*0.6)
        has_upper = False
        has_lower = False
        for c in clothing:
            if bbox_iou(pb, c.bbox) > 0.2:
                cb = c.bbox
                cc = bbox_center(cb)
                if cc[1] < pb[1] + (pb[3]-pb[1])*0.55:
                    has_upper = True
                else:
                    has_lower = True
        has = has_upper  # 以上半身衣物为主判断
        p.label = "已穿工服" if has else "未穿工服"
        if has: persons_with += 1
        else: persons_without += 1
        all_dets.append(p)
    total = len(persons)
    if total == 0: return ClothingDetectResponse(success=True, total_persons=0, persons_with_clothing=0, persons_without_clothing=0, detections=[], alert_level="none", message="未检测到人员")
    elif persons_without == 0: return ClothingDetectResponse(success=True, total_persons=total, persons_with_clothing=persons_with, persons_without_clothing=0, detections=all_dets, alert_level="none", message=f"检测到 {total} 名人员，全部已穿工服")
    else: return ClothingDetectResponse(success=True, total_persons=total, persons_with_clothing=persons_with, persons_without_clothing=persons_without, detections=all_dets, alert_level="critical" if persons_without>=2 else "warning", message=f"检测到 {total} 名人员，其中 {persons_without} 名未穿工服")

# ===================================================================
# 7. 睡岗检测（优化版 - 宽高比 + 位置 + 头部检测辅助）
# ===================================================================
@app.post("/detect/sleeping", response_model=SleepDetectResponse)
async def detect_sleeping(file: UploadFile = File(...)):
    allowed_image(file)
    image, img_w, img_h = await read_image(file)
    result = run_inference(image, conf=PERSON_CONF)
    persons = extract_detections(result, class_filter={"Person"}, min_bbox_ratio=MIN_BBOX_RATIO, img_w=img_w, img_h=img_h)
    heads = extract_detections(result, class_filter={"Human head"})
    sleeping, awake = [], []
    for p in persons:
        ratio = bbox_aspect_ratio(p.bbox)
        y_center = (p.bbox[1] + p.bbox[3]) / 2
        # 睡岗特征：
        # 1. 宽高比 > 1.2（横向躺）+ 位置偏下 > 0.55
        # 2. 宽高比 > 1.6（明显横向）
        # 3. 宽高比 < 0.3（极度窄，可能是侧卧）+ 头部检测确认
        is_sleeping = False
        if ratio > 1.6:
            is_sleeping = True
        elif ratio > 1.2 and y_center > 0.55:
            is_sleeping = True
        elif ratio < 0.3:
            # 检查是否有头部在附近，确认是侧卧而非站立
            has_head = any(bbox_distance(p.bbox, h.bbox) < 0.2 for h in heads)
            is_sleeping = has_head and y_center > 0.5
        p.label = "睡岗" if is_sleeping else "正常"
        (sleeping if is_sleeping else awake).append(p)
    all_dets = sleeping + awake
    n = len(sleeping)
    if n == 0: return SleepDetectResponse(success=True, has_sleeping_person=False, sleeping_count=0, detections=all_dets, alert_level="none", message=f"未检测到睡岗 (检测到 {len(persons)} 名人员)" if persons else "未检测到人员")
    return SleepDetectResponse(success=True, has_sleeping_person=True, sleeping_count=n, detections=all_dets, alert_level="warning" if n==1 else "critical", message=f"检测到 {n} 名人员睡岗")

# ===================================================================
# 8. 跌倒检测（优化版 - 多级宽高比 + 地面位置）
# ===================================================================
@app.post("/detect/fall", response_model=GenericDetectResponse)
async def detect_fall(file: UploadFile = File(...)):
    allowed_image(file)
    image, img_w, img_h = await read_image(file)
    result = run_inference(image, conf=PERSON_CONF)
    persons = extract_detections(result, class_filter={"Person"}, min_bbox_ratio=MIN_BBOX_RATIO, img_w=img_w, img_h=img_h)
    fallen, normal = [], []
    for p in persons:
        ratio = bbox_aspect_ratio(p.bbox)
        yc = (p.bbox[1] + p.bbox[3]) / 2
        bbox_h = p.bbox[3] - p.bbox[1]
        # 跌倒特征（工厂监控视角通常俯拍）：
        # 1. 宽高比 > 1.5 且 bbox 底部在画面下半部
        # 2. 宽高比 > 2.0（几乎横向倒地）
        # 3. 宽高比在 1.0-1.5 但 bbox 高度很小（远处倒地）
        is_fallen = False
        if ratio > 2.0:
            is_fallen = True
        elif ratio > 1.5 and yc > 0.45:
            is_fallen = True
        elif 1.0 < ratio <= 1.5 and bbox_h < 0.25 and yc > 0.6:
            is_fallen = True
        p.label = "跌倒" if is_fallen else "正常"
        (fallen if is_fallen else normal).append(p)
    all_dets = fallen + normal
    n = len(fallen)
    if n == 0: return GenericDetectResponse(success=True, has_target=False, target_count=0, detections=all_dets, alert_level="none", message=f"未检测到跌倒 (检测到 {len(persons)} 名人员)" if persons else "未检测到人员")
    return GenericDetectResponse(success=True, has_target=True, target_count=n, detections=all_dets, alert_level="warning" if n==1 else "critical", message=f"检测到 {n} 名人员跌倒")

# ===================================================================
# 9. 抽烟检测（优化版 - 嘴部区域烟头亮点）
# ===================================================================
@app.post("/detect/smoking", response_model=GenericDetectResponse)
async def detect_smoking(file: UploadFile = File(...)):
    allowed_image(file)
    image, img_w, img_h = await read_image(file)
    img_np = np.array(image)
    result = run_inference(image, conf=0.15)
    persons = extract_detections(result, class_filter={"Person"}, min_bbox_ratio=MIN_BBOX_RATIO, img_w=img_w, img_h=img_h)
    hands = extract_detections(result, class_filter={"Human hand"})
    faces = extract_detections(result, class_filter={"Human face","Human mouth"})
    smoking, normal = [], []
    for p in persons:
        pb = p.bbox
        px1,py1 = int(pb[0]*img_w), int(pb[1]*img_h)
        px2,py2 = int(pb[2]*img_w), int(pb[3]*img_h)
        ph = py2 - py1
        # 嘴部区域：人员 bbox 从上往下 15%-30%
        mouth_y1 = int(py1 + ph * 0.12)
        mouth_y2 = int(py1 + ph * 0.32)
        is_smoking = False
        if mouth_y2 > mouth_y1 and px2 > px1:
            mouth_region = img_np[mouth_y1:mouth_y2, px1:px2, :]
            if mouth_region.size > 0:
                # 烟头特征：小面积高亮（R>200, G>150, B<120 偏橙红）
                r, g, b = mouth_region[:,:,0], mouth_region[:,:,1], mouth_region[:,:,2]
                # 橙色亮点（烟头）
                ember = (r > 200) & (g > 100) & (g < 200) & (b < 130)
                # 灰白烟雾
                smoke = (r > 160) & (g > 150) & (b > 140) & (r < 240)
                ember_ratio = ember.sum() / ember.size if ember.size > 0 else 0
                smoke_ratio = smoke.sum() / smoke.size if smoke.size > 0 else 0
                # 手部靠近脸部
                hand_near_face = any(bbox_distance(pb, h.bbox) < 0.25 for h in hands)
                face_detected = any(bbox_iou(pb, f.bbox) > 0.02 for f in faces)
                is_smoking = (ember_ratio > 0.003 and hand_near_face) or (smoke_ratio > 0.08 and face_detected)
        p.label = "抽烟" if is_smoking else "正常"
        (smoking if is_smoking else normal).append(p)
    all_dets = smoking + normal
    n = len(smoking)
    if n == 0: return GenericDetectResponse(success=True, has_target=False, target_count=0, detections=all_dets, alert_level="none", message=f"未检测到抽烟 (检测到 {len(persons)} 名人员)" if persons else "未检测到人员")
    return GenericDetectResponse(success=True, has_target=True, target_count=n, detections=all_dets, alert_level="warning" if n==1 else "critical", message=f"检测到 {n} 名人员抽烟")

# ===================================================================
# 10. 使用手机检测（优化版 - 手机在手部区域判断）
# ===================================================================
@app.post("/detect/phone", response_model=GenericDetectResponse)
async def detect_phone(file: UploadFile = File(...)):
    allowed_image(file)
    image, img_w, img_h = await read_image(file)
    result = run_inference(image, conf=0.15)
    persons = extract_detections(result, class_filter={"Person"}, min_bbox_ratio=MIN_BBOX_RATIO, img_w=img_w, img_h=img_h)
    phones = extract_detections(result, class_filter={"Mobile phone","Corded phone","Telephone","Tablet computer","Ipod"})
    hands = extract_detections(result, class_filter={"Human hand"})
    phone_p, normal_p = [], []
    for p in persons:
        pb = p.bbox
        has = False
        for ph in phones:
            phc = bbox_center(ph.bbox)
            # 手机必须在人员区域内或紧邻
            if bbox_iou(pb, ph.bbox) > 0.03:
                has = True; break
            # 或者在人员附近 + 手部也在附近
            if bbox_distance(pb, ph.bbox) < 0.25:
                # 确认手机在手部附近
                for hd in hands:
                    if bbox_distance(ph.bbox, hd.bbox) < 0.15:
                        has = True; break
                if has: break
        p.label = "使用手机" if has else "正常"
        (phone_p if has else normal_p).append(p)
    all_dets = phone_p + normal_p
    n = len(phone_p)
    if n == 0: return GenericDetectResponse(success=True, has_target=False, target_count=0, detections=all_dets, alert_level="none", message=f"未检测到使用手机 (检测到 {len(persons)} 名人员)" if persons else "未检测到人员")
    return GenericDetectResponse(success=True, has_target=True, target_count=n, detections=all_dets, alert_level="warning" if n==1 else "critical", message=f"检测到 {n} 名人员使用手机")

# ===================================================================
# 11. 打架检测（优化版 - 相对距离 + 重叠度）
# ===================================================================
@app.post("/detect/fight", response_model=FightDetectResponse)
async def detect_fight(file: UploadFile = File(...)):
    allowed_image(file)
    image, img_w, img_h = await read_image(file)
    result = run_inference(image, conf=PERSON_CONF)
    persons = extract_detections(result, class_filter={"Person"}, min_bbox_ratio=MIN_BBOX_RATIO, img_w=img_w, img_h=img_h)
    fighting, groups = set(), 0
    n = len(persons)
    for i in range(n):
        for j in range(i+1, n):
            b1, b2 = persons[i].bbox, persons[j].bbox
            dist = bbox_distance(b1, b2)
            iou = bbox_iou(b1, b2)
            # 用人员 bbox 宽度的相对距离判断
            avg_w = ((b1[2]-b1[0]) + (b2[2]-b2[0])) / 2
            # 两人距离小于平均宽度的 0.8 倍，或 IoU > 0.1
            if dist < avg_w * 0.8 or iou > 0.1:
                fighting.add(i); fighting.add(j); groups += 1
    has = len(fighting) >= 2
    for i,p in enumerate(persons): p.label = "打架" if i in fighting else "正常"
    if not has: return FightDetectResponse(success=True, has_fight=False, person_count=n, fight_groups=0, detections=persons, alert_level="none", message=f"未检测到打架 (检测到 {n} 名人员)" if n else "未检测到人员")
    return FightDetectResponse(success=True, has_fight=True, person_count=n, fight_groups=groups, detections=persons, alert_level="critical", message=f"检测到疑似打架，涉及 {len(fighting)} 名人员")

# ===================================================================
# 12. 电瓶车检测
# ===================================================================
@app.post("/detect/e-bike", response_model=VehicleDetectResponse)
async def detect_ebike(file: UploadFile = File(...)):
    allowed_image(file)
    image, _, _ = await read_image(file)
    dets = extract_detections(run_inference(image, conf=0.3), class_filter={"Bicycle","Motorcycle"})
    for d in dets: d.label = "电瓶车"
    n = len(dets)
    return VehicleDetectResponse(success=True, has_vehicle=n>0, vehicle_count=n, detections=dets, alert_level="none" if n==0 else ("warning" if n==1 else "critical"), message=f"检测到 {n} 辆电瓶车" if n else "未检测到电瓶车")

# ===================================================================
# 13. 货车检测
# ===================================================================
@app.post("/detect/truck", response_model=VehicleDetectResponse)
async def detect_truck(file: UploadFile = File(...)):
    allowed_image(file)
    image, _, _ = await read_image(file)
    dets = extract_detections(run_inference(image, conf=0.3), class_filter={"Truck"})
    for d in dets: d.label = "货车"
    n = len(dets)
    return VehicleDetectResponse(success=True, has_vehicle=n>0, vehicle_count=n, detections=dets, alert_level="none" if n==0 else ("warning" if n==1 else "critical"), message=f"检测到 {n} 辆货车" if n else "未检测到货车")

# ===================================================================
# 14. 小汽车检测
# ===================================================================
@app.post("/detect/car", response_model=VehicleDetectResponse)
async def detect_car(file: UploadFile = File(...)):
    allowed_image(file)
    image, _, _ = await read_image(file)
    dets = extract_detections(run_inference(image, conf=0.3), class_filter={"Car"})
    for d in dets: d.label = "小汽车"
    n = len(dets)
    return VehicleDetectResponse(success=True, has_vehicle=n>0, vehicle_count=n, detections=dets, alert_level="none" if n==0 else ("warning" if n==1 else "critical"), message=f"检测到 {n} 辆小汽车" if n else "未检测到小汽车")

# ===================================================================
# 统一 Demo 页面
# ===================================================================
DEMO_HTML = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CV Detector - 视觉检测平台</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px}
.container{width:100%;max-width:800px;background:#1e293b;border-radius:16px;padding:28px;box-shadow:0 25px 50px rgba(0,0,0,.4)}
h1{font-size:22px;font-weight:700;text-align:center;margin-bottom:4px;background:linear-gradient(135deg,#38bdf8,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.subtitle{text-align:center;color:#94a3b8;font-size:13px;margin-bottom:20px}
.api-selector{display:flex;flex-wrap:wrap;gap:6px;justify-content:center;margin-bottom:20px}
.api-btn{padding:6px 14px;border-radius:20px;font-size:12px;font-weight:600;cursor:pointer;border:1px solid #475569;background:#1e293b;color:#94a3b8;transition:all .2s}
.api-btn:hover{border-color:#38bdf8;color:#38bdf8}
.api-btn.active{background:linear-gradient(135deg,#38bdf8,#818cf8);color:#fff;border-color:transparent}
.dropzone{border:2px dashed #475569;border-radius:12px;padding:36px 20px;text-align:center;cursor:pointer;transition:all .2s;margin-bottom:16px}
.dropzone:hover,.dropzone.dragover{border-color:#38bdf8;background:rgba(56,189,248,.05)}
.dropzone-icon{font-size:36px;margin-bottom:10px}
.dropzone-text{color:#94a3b8;font-size:14px}
.dropzone-hint{color:#64748b;font-size:12px;margin-top:4px}
#fileInput{display:none}
.preview-area{display:none;position:relative;border-radius:12px;overflow:hidden;margin-bottom:16px;background:#0f172a}
.preview-area.show{display:block}
.preview-area img{width:100%;display:block;max-height:440px;object-fit:contain}
.preview-area canvas{position:absolute;top:0;left:0;width:100%;height:100%}
.loading{display:none;text-align:center;padding:10px;color:#38bdf8;font-size:13px}
.loading.show{display:block}
.spinner{display:inline-block;width:16px;height:16px;border:2px solid rgba(56,189,248,.3);border-top-color:#38bdf8;border-radius:50%;animation:spin .6s linear infinite;vertical-align:middle;margin-right:6px}
@keyframes spin{to{transform:rotate(360deg)}}
.result{display:none;padding:16px;border-radius:10px;margin-bottom:16px;font-size:13px}
.result.show{display:block}
.result.none{background:rgba(74,222,128,.1);border:1px solid rgba(74,222,128,.3)}
.result.warning{background:rgba(251,191,36,.1);border:1px solid rgba(251,191,36,.3)}
.result.critical{background:rgba(248,113,113,.1);border:1px solid rgba(248,113,113,.3)}
.result.error{background:rgba(148,163,184,.1);border:1px solid rgba(148,163,184,.2)}
.result-icon{font-size:40px;text-align:center;margin-bottom:6px}
.result-title{font-size:17px;font-weight:700;text-align:center;margin-bottom:4px}
.result-detail{font-size:13px;color:#94a3b8;text-align:center;margin-top:6px}
.detection-item{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;background:rgba(255,255,255,.05);border-radius:6px;margin:3px;font-size:12px}
.detection-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.btn-row{display:flex;gap:10px}
.btn{flex:1;padding:11px;border:none;border-radius:10px;font-size:14px;font-weight:600;cursor:pointer;transition:all .2s}
.btn-detect{background:linear-gradient(135deg,#38bdf8,#818cf8);color:#fff}
.btn-detect:hover{opacity:.9;transform:translateY(-1px)}
.btn-detect:disabled{opacity:.4;cursor:not-allowed;transform:none}
.btn-reset{background:#334155;color:#cbd5e1}
.btn-reset:hover{background:#475569}
.footer{text-align:center;color:#475569;font-size:11px;margin-top:16px}
.summary-stats{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin-top:8px}
.stat-badge{padding:4px 10px;border-radius:12px;font-size:12px;font-weight:600}
.stat-good{background:rgba(74,222,128,.15);color:#4ade80}
.stat-bad{background:rgba(248,113,113,.15);color:#f87171}
</style>
</head>
<body>
<div class="container">
<h1>🔍 CV Detector 视觉检测平台</h1>
<p class="subtitle">选择检测类型 → 上传图片 → AI 自动分析</p>

<div class="api-selector" id="apiSelector">
<button class="api-btn active" data-api="/detect">🐾 动物检测</button>
<button class="api-btn" data-api="/detect/person">👤 人员入侵检测</button>
<button class="api-btn" data-api="/detect/helmet">⛑️ 未佩戴安全帽检测</button>
<button class="api-btn" data-api="/detect/reflective-vest">🦺 未穿戴反光衣检测</button>
<button class="api-btn" data-api="/detect/uniform">👔 未穿工服检测</button>
<button class="api-btn" data-api="/detect/sleeping">😴 睡岗检测</button>
<button class="api-btn" data-api="/detect/fall">🤸 跌倒检测</button>
<button class="api-btn" data-api="/detect/smoking">🚬 抽烟检测</button>
<button class="api-btn" data-api="/detect/phone">📱 使用手机检测</button>
<button class="api-btn" data-api="/detect/fight">👊 打架检测</button>
<button class="api-btn" data-api="/detect/e-bike">🛵 电瓶车检测</button>
<button class="api-btn" data-api="/detect/truck">🚛 货车检测</button>
<button class="api-btn" data-api="/detect/car">🚗 小汽车检测</button>
</div>

<div class="dropzone" id="dropzone">
<div class="dropzone-icon">📁</div>
<div class="dropzone-text">点击选择图片或拖拽到此处</div>
<div class="dropzone-hint">支持 JPG / PNG / BMP / WEBP</div>
<input type="file" id="fileInput" accept="image/jpeg,image/png,image/bmp,image/webp">
</div>

<div class="preview-area" id="previewArea">
<img id="previewImg" alt="preview">
<canvas id="bboxCanvas"></canvas>
</div>

<div class="loading" id="loading"><span class="spinner"></span>正在检测中...</div>
<div class="result" id="result"></div>

<div class="btn-row">
<button class="btn btn-detect" id="btnDetect" disabled>🔍 开始检测</button>
<button class="btn btn-detect" id="btnDetectAll" disabled style="background:linear-gradient(135deg,#a78bfa,#f472b6)">🚀 一键全检</button>
<button class="btn btn-reset" id="btnReset">🔄 重新选择</button>
</div>

</div>

<script>
let selectedFile = null;
let currentApi = '/detect';

const COLOR_MAP = {
  'Person':'#38bdf8','Snake':'#4ade80','Lizard':'#fbbf24','Cat':'#f472b6','Dog':'#a78bfa','Wild boar':'#fb923c',
  '已佩戴安全帽':'#4ade80','未佩戴安全帽':'#f87171',
  '已穿戴反光衣':'#4ade80','未穿戴反光衣':'#f87171',
  '已穿工服':'#4ade80','未穿工服':'#f87171',
  '睡岗':'#f87171','正常':'#4ade80',
  '跌倒':'#f87171',
  '抽烟':'#f87171',
  '使用手机':'#fbbf24',
  '打架':'#f87171',
  '电瓶车':'#fb923c','货车':'#a78bfa','小汽车':'#38bdf8',
  '人员':'#fb923c'
};

const apiSelector = document.getElementById('apiSelector');
apiSelector.addEventListener('click', (e) => {
  if (e.target.classList.contains('api-btn')) {
    apiSelector.querySelectorAll('.api-btn').forEach(b => b.classList.remove('active'));
    e.target.classList.add('active');
    currentApi = e.target.dataset.api;
  }
});

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const previewArea = document.getElementById('previewArea');
const previewImg = document.getElementById('previewImg');
const bboxCanvas = document.getElementById('bboxCanvas');
const loading = document.getElementById('loading');
const result = document.getElementById('result');
const btnDetect = document.getElementById('btnDetect');
const btnDetectAll = document.getElementById('btnDetectAll');
const btnReset = document.getElementById('btnReset');

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', e => {
  e.preventDefault(); dropzone.classList.remove('dragover');
  if (e.dataTransfer.files.length>0) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files.length>0) handleFile(fileInput.files[0]); });

function handleFile(file) {
  if (!file.type.match(/image\/(jpeg|png|bmp|webp)/)) { alert('不支持该格式'); return; }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    previewArea.classList.add('show');
    dropzone.style.display = 'none';
    btnDetect.disabled = false;
    btnDetectAll.disabled = false;
    result.classList.remove('show','none','warning','critical','error');
    result.innerHTML = '';
    clearCanvas();
  };
  reader.readAsDataURL(file);
}

// 所有接口列表
const ALL_APIS = [
  '/detect', '/detect/person', '/detect/helmet', '/detect/reflective-vest',
  '/detect/uniform', '/detect/sleeping', '/detect/fall', '/detect/smoking',
  '/detect/phone', '/detect/fight', '/detect/e-bike', '/detect/truck', '/detect/car'
];

const API_LABELS = {
  '/detect':'🐾 动物检测', '/detect/person':'👤 人员入侵检测', '/detect/helmet':'⛑️ 未佩戴安全帽检测',
  '/detect/reflective-vest':'🦺 未穿戴反光衣检测', '/detect/uniform':'👔 未穿工服检测', '/detect/sleeping':'😴 睡岗检测',
  '/detect/fall':'🤸 跌倒检测', '/detect/smoking':'🚬 抽烟检测', '/detect/phone':'📱 使用手机检测',
  '/detect/fight':'👊 打架检测', '/detect/e-bike':'🛵 电瓶车检测', '/detect/truck':'🚛 货车检测', '/detect/car':'🚗 小汽车检测'
};

function getSummary(data, api) {
  if (!data.success) return { icon:'❌', text:'失败', color:'#94a3b8', level:'error' };
  if (api === '/detect') {
    if (data.has_animal) return { icon:'🔍', text:data.count+' 个目标', color:'#4ade80', level:'none' };
    return { icon:'✅', text:'无目标', color:'#94a3b8', level:'none' };
  }
  if (api === '/detect/person') {
    if (data.has_person) return { icon:'⚠️', text:data.person_count+' 人', color:data.alert_level==='critical'?'#f87171':'#fbbf24', level:data.alert_level };
    return { icon:'✅', text:'无人', color:'#4ade80', level:'none' };
  }
  if (api === '/detect/helmet' || api === '/detect/reflective-vest' || api === '/detect/uniform') {
    const bad = api==='/detect/helmet'?data.persons_without_helmet:(api==='/detect/reflective-vest'||api==='/detect/uniform'?data.persons_without_clothing:0);
    if (data.total_persons===0) return { icon:'ℹ️', text:'无人', color:'#94a3b8', level:'none' };
    if (bad===0) return { icon:'✅', text:data.total_persons+'人合规', color:'#4ade80', level:'none' };
    return { icon:'⚠️', text:bad+'/'+data.total_persons+'人违规', color:'#f87171', level:'warning' };
  }
  if (api === '/detect/sleeping') {
    if (data.has_sleeping_person) return { icon:'😴', text:data.sleeping_count+'人睡岗', color:'#f87171', level:'critical' };
    return { icon:'✅', text:'正常', color:'#4ade80', level:'none' };
  }
  if (api === '/detect/fall' || api === '/detect/smoking' || api === '/detect/phone') {
    if (data.has_target) return { icon:'⚠️', text:data.target_count+'人', color:'#f87171', level:'warning' };
    return { icon:'✅', text:'正常', color:'#4ade80', level:'none' };
  }
  if (api === '/detect/fight') {
    if (data.has_fight) return { icon:'👊', text:'疑似打架', color:'#f87171', level:'critical' };
    return { icon:'✅', text:'正常', color:'#4ade80', level:'none' };
  }
  if (api === '/detect/e-bike' || api === '/detect/truck' || api === '/detect/car') {
    if (data.has_vehicle) return { icon:'🚨', text:data.vehicle_count+'辆', color:'#fbbf24', level:'warning' };
    return { icon:'✅', text:'未检测到', color:'#4ade80', level:'none' };
  }
  return { icon:'?', text:'-', color:'#94a3b8', level:'none' };
}

btnDetectAll.addEventListener('click', async () => {
  if (!selectedFile) return;
  btnDetect.disabled = true;
  btnDetectAll.disabled = true;
  loading.classList.add('show');
  result.classList.remove('show','none','warning','critical','error');
  clearCanvas();

  // 显示进度
  result.className = 'result show none';
  result.innerHTML = '<div class="result-icon">⏳</div><div class="result-title" style="color:#38bdf8">正在全量检测中...</div><div class="result-detail" id="allProgress">准备调用 13 个接口</div>';

  const allResults = [];
  let completed = 0;

  // 并发调用所有接口（每批3个，避免服务器过载）
  const batchSize = 3;
  for (let i = 0; i < ALL_APIS.length; i += batchSize) {
    const batch = ALL_APIS.slice(i, i + batchSize);
    const promises = batch.map(async (api) => {
      const fd = new FormData(); fd.append('file', selectedFile);
      try {
        const resp = await fetch(api, { method:'POST', body:fd });
        const data = await resp.json();
        return { api, data, ok: true };
      } catch(err) {
        return { api, data:null, ok: false, error: err.message };
      }
    });
    const batchResults = await Promise.all(promises);
    allResults.push(...batchResults);
    completed += batch.length;
    document.getElementById('allProgress').textContent = `已完成 ${completed}/${ALL_APIS.length} 个接口`;
  }

  // 渲染汇总结果
  let warnings = 0, criticals = 0;
  let html = '<div class="result-icon">📊</div>';
  html += '<div class="result-title" style="color:#e2e8f0">全量检测报告</div>';
  html += '<div class="result-detail" style="margin-bottom:12px">共调用 13 个接口，结果如下：</div>';
  html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;text-align:left;max-height:360px;overflow-y:auto;padding:0 8px">';

  allResults.forEach(r => {
    const label = API_LABELS[r.api] || r.api;
    if (r.ok) {
      const s = getSummary(r.data, r.api);
      if (s.level === 'warning') warnings++;
      if (s.level === 'critical') criticals++;
      html += '<div style="display:flex;align-items:center;justify-content:space-between;padding:6px 10px;background:rgba(255,255,255,.03);border-radius:8px;font-size:12px">';
      html += '<span>'+label+'</span>';
      html += '<span style="color:'+s.color+';font-weight:600">'+s.icon+' '+s.text+'</span>';
      html += '</div>';
    } else {
      html += '<div style="display:flex;align-items:center;justify-content:space-between;padding:6px 10px;background:rgba(255,255,255,.03);border-radius:8px;font-size:12px">';
      html += '<span>'+label+'</span>';
      html += '<span style="color:#f87171">❌ 失败</span>';
      html += '</div>';
    }
  });

  html += '</div>';

  // 汇总统计
  let level = 'none', titleColor = '#4ade80', summaryIcon = '✅', summaryTitle = '全部正常';
  if (criticals > 0) { level = 'critical'; titleColor = '#f87171'; summaryIcon = '🚨'; summaryTitle = criticals+' 项危险'; }
  else if (warnings > 0) { level = 'warning'; titleColor = '#fbbf24'; summaryIcon = '⚠️'; summaryTitle = warnings+' 项警告'; }

  html += '<div class="result-detail" style="margin-top:10px">';
  html += '<span style="font-size:16px">'+summaryIcon+'</span> ';
  html += '<span style="color:'+titleColor+';font-weight:700">'+summaryTitle+'</span>';
  if (warnings>0 || criticals>0) html += ' <span style="color:#94a3b8">（警告 '+warnings+' 项，危险 '+criticals+' 项）</span>';
  html += '</div>';

  result.className = 'result show ' + level;
  result.innerHTML = html;

  loading.classList.remove('show');
  btnDetect.disabled = false;
  btnDetectAll.disabled = false;
});

btnDetect.addEventListener('click', async () => {
  if (!selectedFile) return;
  btnDetect.disabled = true;
  loading.classList.add('show');
  result.classList.remove('show','none','warning','critical','error');
  clearCanvas();
  const fd = new FormData(); fd.append('file', selectedFile);
  try {
    const resp = await fetch(currentApi, { method:'POST', body:fd });
    const data = await resp.json();
    renderResult(data);
  } catch(err) {
    result.className = 'result show error';
    result.innerHTML = '<div class="result-icon">❌</div><div class="result-title">请求失败</div><div class="result-detail">'+err.message+'</div>';
  } finally {
    loading.classList.remove('show');
    btnDetect.disabled = false;
  }
});

function renderResult(data) {
  if (!data.success) {
    result.className = 'result show error';
    result.innerHTML = '<div class="result-icon">❌</div><div class="result-title">检测失败</div>';
    return;
  }

  // 判断告警级别
  let level = 'none';
  if (data.alert_level) level = data.alert_level;
  else if (data.has_animal === false && data.has_person === false && data.has_fight === false && data.has_vehicle === false && data.has_target === false && data.has_sleeping_person === false) level = 'none';
  else level = 'warning';

  let icon = '✅', title = '安全', titleColor = '#4ade80', detail = '';

  // 根据接口类型渲染不同结果
  if (currentApi === '/detect') {
    if (data.has_animal) {
      icon = '🔍'; title = data.message; titleColor = '#4ade80'; level = 'none';
    } else {
      icon = '✅'; title = '未检测到目标'; titleColor = '#94a3b8';
    }
  } else if (currentApi === '/detect/person') {
    if (data.has_person) {
      icon = data.alert_level==='critical'?'🚨':'⚠️';
      title = data.alert_level==='critical'?'危险 · 多人入侵':'警告 · 检测到人员';
      titleColor = data.alert_level==='critical'?'#f87171':'#fbbf24';
      detail = '检测到 '+data.person_count+' 名人员';
    } else {
      icon = '✅'; title = '安全 · 未检测到人员'; titleColor = '#4ade80';
    }
  } else if (currentApi === '/detect/helmet') {
    if (data.total_persons === 0) { icon='ℹ️'; title='未检测到人员'; titleColor='#94a3b8'; }
    else if (data.persons_without_helmet === 0) { icon='✅'; title='全部已佩戴安全帽'; titleColor='#4ade80'; detail=data.total_persons+' 名人员均合规'; }
    else { icon='⚠️'; title=data.persons_without_helmet+' 名未佩戴安全帽'; titleColor='#f87171'; detail='共 '+data.total_persons+' 名人员'; }
  } else if (currentApi === '/detect/reflective-vest') {
    if (data.total_persons === 0) { icon='ℹ️'; title='未检测到人员'; titleColor='#94a3b8'; }
    else if (data.persons_without_clothing === 0) { icon='✅'; title='全部已穿戴反光衣'; titleColor='#4ade80'; detail=data.total_persons+' 名人员均合规'; }
    else { icon='⚠️'; title=data.persons_without_clothing+' 名未穿戴反光衣'; titleColor='#f87171'; detail='共 '+data.total_persons+' 名人员'; }
  } else if (currentApi === '/detect/uniform') {
    if (data.total_persons === 0) { icon='ℹ️'; title='未检测到人员'; titleColor='#94a3b8'; }
    else if (data.persons_without_clothing === 0) { icon='✅'; title='全部已穿工服'; titleColor='#4ade80'; detail=data.total_persons+' 名人员均合规'; }
    else { icon='⚠️'; title=data.persons_without_clothing+' 名未穿工服'; titleColor='#f87171'; detail='共 '+data.total_persons+' 名人员'; }
  } else if (currentApi === '/detect/sleeping') {
    if (data.has_sleeping_person) { icon='😴'; title='检测到 '+data.sleeping_count+' 名睡岗'; titleColor='#f87171'; level='critical'; }
    else { icon='✅'; title='未检测到睡岗'; titleColor='#4ade80'; }
  } else if (currentApi === '/detect/fall') {
    if (data.has_target) { icon='🤸'; title='检测到 '+data.target_count+' 名跌倒'; titleColor='#f87171'; level='critical'; }
    else { icon='✅'; title='未检测到跌倒'; titleColor='#4ade80'; }
  } else if (currentApi === '/detect/smoking') {
    if (data.has_target) { icon='🚬'; title='检测到 '+data.target_count+' 名抽烟'; titleColor='#f87171'; level='critical'; }
    else { icon='✅'; title='未检测到抽烟'; titleColor='#4ade80'; }
  } else if (currentApi === '/detect/phone') {
    if (data.has_target) { icon='📱'; title='检测到 '+data.target_count+' 名使用手机'; titleColor='#fbbf24'; level='warning'; }
    else { icon='✅'; title='未检测到使用手机'; titleColor='#4ade80'; }
  } else if (currentApi === '/detect/fight') {
    if (data.has_fight) { icon='👊'; title='检测到疑似打架'; titleColor='#f87171'; level='critical'; detail='涉及 '+data.person_count+' 名人员'; }
    else { icon='✅'; title='未检测到打架'; titleColor='#4ade80'; detail='检测到 '+data.person_count+' 名人员'+(data.person_count?'':'未检测到人员'); }
  } else if (currentApi === '/detect/e-bike' || currentApi === '/detect/truck' || currentApi === '/detect/car') {
    if (data.has_vehicle) { icon='🚨'; title='检测到 '+data.vehicle_count+' 辆'; titleColor='#fbbf24'; level='warning'; }
    else { icon='✅'; title='未检测到目标车辆'; titleColor='#4ade80'; }
  }

  result.className = 'result show ' + level;
  let html = '<div class="result-icon">'+icon+'</div>';
  html += '<div class="result-title" style="color:'+titleColor+'">'+title+'</div>';
  if (detail) html += '<div class="result-detail">'+detail+'</div>';
  if (data.detections && data.detections.length > 0) {
    html += '<div class="result-detail">';
    data.detections.forEach(d => {
      const color = COLOR_MAP[d.label] || COLOR_MAP[d.class_name] || '#fff';
      html += '<span class="detection-item"><span class="detection-dot" style="background:'+color+'"></span>'+d.label+' '+(d.confidence*100).toFixed(1)+'%</span>';
    });
    html += '</div>';
  }
  result.innerHTML = html;

  if (data.detections && data.detections.length > 0) drawBBoxes(data.detections);
}

btnReset.addEventListener('click', () => {
  selectedFile = null; fileInput.value = '';
  previewArea.classList.remove('show'); previewImg.src = '';
  dropzone.style.display = ''; btnDetect.disabled = true; btnDetectAll.disabled = true;
  result.classList.remove('show','none','warning','critical','error');
  result.innerHTML = ''; clearCanvas();
});

function drawBBoxes(dets) {
  const img = previewImg;
  const canvas = bboxCanvas;
  canvas.width = img.offsetWidth; canvas.height = img.offsetHeight;
  const ctx = canvas.getContext('2d');
  dets.forEach(d => {
    const [x1,y1,x2,y2] = d.bbox;
    const px=x1*canvas.width, py=y1*canvas.height, pw=(x2-x1)*canvas.width, ph=(y2-y1)*canvas.height;
    const color = COLOR_MAP[d.label] || COLOR_MAP[d.class_name] || '#fb923c';
    ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.strokeRect(px,py,pw,ph);
    ctx.fillStyle = color; ctx.font = 'bold 13px -apple-system,sans-serif';
    const label = d.label+' '+(d.confidence*100).toFixed(0)+'%';
    const tm = ctx.measureText(label); const lw = tm.width+8, lh=20;
    ctx.fillRect(px, py-lh, lw, lh);
    ctx.fillStyle = '#0f172a'; ctx.fillText(label, px+4, py-5);
  });
}

function clearCanvas() {
  const ctx = bboxCanvas.getContext('2d');
  ctx.clearRect(0, 0, bboxCanvas.width, bboxCanvas.height);
}
</script>
</body>
</html>'''

@app.get("/demo", response_class=HTMLResponse)
def demo_page():
    return DEMO_HTML

@app.get("/demo/person", response_class=HTMLResponse)
def demo_person_page():
    return DEMO_HTML

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
