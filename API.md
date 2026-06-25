# 🔍 CV Detector API 文档

> 模型：YOLOv8s Open Images V7（600 类别）  
> 版本：2.0.0  
> 更新时间：2026-06-25

---

## 服务地址

| 方式 | 地址 |
|------|------|
| HTTPS | https://gtt666.cn |
| HTTP（自动跳转 HTTPS） | http://gtt666.cn |

---

## 接口总览

| # | 接口 | 方法 | 功能 |
|---|------|------|------|
| - | `/` | GET | 服务信息 |
| - | `/health` | GET | 健康检查 |
| 1 | `/detect` | POST | 动物 + 人类检测 |
| 2 | `/detect/batch` | POST | 批量检测（最多10张） |
| 3 | `/detect/person` | POST | 人员入侵检测 |
| 4 | `/detect/helmet` | POST | 未佩戴安全帽检测 |
| 5 | `/detect/reflective-vest` | POST | 未穿戴反光衣检测 |
| 6 | `/detect/uniform` | POST | 未穿工服检测 |
| 7 | `/detect/sleeping` | POST | 睡岗检测 |
| 8 | `/detect/fall` | POST | 跌倒检测 |
| 9 | `/detect/smoking` | POST | 抽烟检测 |
| 10 | `/detect/phone` | POST | 使用手机检测 |
| 11 | `/detect/fight` | POST | 打架检测 |
| 12 | `/detect/e-bike` | POST | 电瓶车检测 |
| 13 | `/detect/truck` | POST | 货车检测 |
| 14 | `/detect/car` | POST | 小汽车检测 |

---

## 通用说明

**所有 POST 接口：**
- Content-Type: `multipart/form-data`
- 参数: `file` — 图片文件（JPG/PNG/BMP/WEBP）
- 返回字段均包含 `success`、`alert_level`（none/warning/critical）、`message`

**告警级别：**
| 级别 | 含义 |
|------|------|
| `none` | 安全，未检测到异常 |
| `warning` | 警告，检测到 1 个异常目标 |
| `critical` | 危险，检测到 2 个及以上异常目标 |

---

## 1. 动物 + 人类检测

**POST** `/detect`

```bash
curl -X POST https://gtt666.cn/detect -F "file=@image.jpg"
```

返回字段：`success`, `has_animal`, `detections`, `count`, `message`

```json
{
  "success": true,
  "has_animal": true,
  "detections": [
    {"class_name": "Cat", "label": "猫", "confidence": 0.89, "bbox": [0.12, 0.34, 0.56, 0.78]}
  ],
  "count": 1,
  "message": "检测到目标: 猫(0.89)"
}
```

---

## 2. 批量检测

**POST** `/detect/batch`

```bash
curl -X POST https://gtt666.cn/detect/batch -F "files=@a.jpg" -F "files=@b.jpg"
```

返回 `list[DetectResponse]`，最多 10 张。

---

## 3. 人员入侵检测

**POST** `/detect/person`

返回字段：`success`, `has_person`, `person_count`, `detections`, `alert_level`, `message`

```json
{
  "success": true,
  "has_person": true,
  "person_count": 1,
  "detections": [{"class_name": "Person", "label": "人员", "confidence": 0.75, "bbox": [0.36, 0.24, 0.64, 0.92]}],
  "alert_level": "warning",
  "message": "检测到 1 名人员"
}
```

---

## 4. 未佩戴安全帽检测 🆕

**POST** `/detect/helmet`

返回字段：`success`, `total_persons`, `persons_with_helmet`, `persons_without_helmet`, `detections`, `alert_level`, `message`

```json
{
  "success": true,
  "total_persons": 3,
  "persons_with_helmet": 2,
  "persons_without_helmet": 1,
  "detections": [
    {"class_name": "Person", "label": "已佩戴安全帽", "confidence": 0.85, "bbox": [...]},
    {"class_name": "Person", "label": "已佩戴安全帽", "confidence": 0.78, "bbox": [...]},
    {"class_name": "Person", "label": "未佩戴安全帽", "confidence": 0.72, "bbox": [...]}
  ],
  "alert_level": "warning",
  "message": "检测到 3 名人员，其中 1 名未佩戴安全帽"
}
```

---

## 5. 未穿戴反光衣检测 🆕

**POST** `/detect/reflective-vest`

返回字段：`success`, `total_persons`, `persons_with_clothing`, `persons_without_clothing`, `detections`, `alert_level`, `message`

```json
{
  "success": true,
  "total_persons": 2,
  "persons_with_clothing": 1,
  "persons_without_clothing": 1,
  "detections": [
    {"class_name": "Person", "label": "已穿戴反光衣", "confidence": 0.81, "bbox": [...]},
    {"class_name": "Person", "label": "未穿戴反光衣", "confidence": 0.76, "bbox": [...]}
  ],
  "alert_level": "warning",
  "message": "检测到 2 名人员，其中 1 名未穿戴反光衣"
}
```

---

## 6. 未穿工服检测 🆕

**POST** `/detect/uniform`

返回字段同反光衣检测。

```json
{
  "success": true,
  "total_persons": 1,
  "persons_with_clothing": 0,
  "persons_without_clothing": 1,
  "detections": [{"class_name": "Person", "label": "未穿工服", "confidence": 0.73, "bbox": [...]}],
  "alert_level": "warning",
  "message": "检测到 1 名人员，其中 1 名未穿工服"
}
```

---

## 7. 睡岗检测 🆕

**POST** `/detect/sleeping`

返回字段：`success`, `has_sleeping_person`, `sleeping_count`, `detections`, `alert_level`, `message`

```json
{
  "success": true,
  "has_sleeping_person": true,
  "sleeping_count": 1,
  "detections": [{"class_name": "Person", "label": "睡岗", "confidence": 0.68, "bbox": [...]}],
  "alert_level": "warning",
  "message": "检测到 1 名人员睡岗"
}
```

---

## 8. 跌倒检测 🆕

**POST** `/detect/fall`

返回字段：`success`, `has_target`, `target_count`, `detections`, `alert_level`, `message`

```json
{
  "success": true,
  "has_target": true,
  "target_count": 1,
  "detections": [{"class_name": "Person", "label": "跌倒", "confidence": 0.71, "bbox": [...]}],
  "alert_level": "warning",
  "message": "检测到 1 名人员跌倒"
}
```

---

## 9. 抽烟检测 🆕

**POST** `/detect/smoking`

返回字段：`success`, `has_target`, `target_count`, `detections`, `alert_level`, `message`

```json
{
  "success": true,
  "has_target": true,
  "target_count": 1,
  "detections": [{"class_name": "Person", "label": "抽烟", "confidence": 0.65, "bbox": [...]}],
  "alert_level": "warning",
  "message": "检测到 1 名人员抽烟"
}
```

---

## 10. 使用手机检测 🆕

**POST** `/detect/phone`

返回字段：`success`, `has_target`, `target_count`, `detections`, `alert_level`, `message`

```json
{
  "success": true,
  "has_target": true,
  "target_count": 1,
  "detections": [{"class_name": "Person", "label": "使用手机", "confidence": 0.70, "bbox": [...]}],
  "alert_level": "warning",
  "message": "检测到 1 名人员使用手机"
}
```

---

## 11. 打架检测 🆕

**POST** `/detect/fight`

返回字段：`success`, `has_fight`, `person_count`, `fight_groups`, `detections`, `alert_level`, `message`

```json
{
  "success": true,
  "has_fight": true,
  "person_count": 3,
  "fight_groups": 2,
  "detections": [
    {"class_name": "Person", "label": "打架", "confidence": 0.82, "bbox": [...]},
    {"class_name": "Person", "label": "打架", "confidence": 0.79, "bbox": [...]},
    {"class_name": "Person", "label": "正常", "confidence": 0.74, "bbox": [...]}
  ],
  "alert_level": "critical",
  "message": "检测到疑似打架行为，涉及 2 名人员"
}
```

---

## 12. 电瓶车检测 🆕

**POST** `/detect/e-bike`

返回字段：`success`, `has_vehicle`, `vehicle_count`, `detections`, `alert_level`, `message`

```json
{
  "success": true,
  "has_vehicle": true,
  "vehicle_count": 2,
  "detections": [
    {"class_name": "Bicycle", "label": "电瓶车", "confidence": 0.78, "bbox": [...]},
    {"class_name": "Motorcycle", "label": "电瓶车", "confidence": 0.72, "bbox": [...]}
  ],
  "alert_level": "critical",
  "message": "检测到 2 辆电瓶车"
}
```

---

## 13. 货车检测 🆕

**POST** `/detect/truck`

返回字段同电瓶车检测。

```json
{
  "success": true,
  "has_vehicle": true,
  "vehicle_count": 1,
  "detections": [{"class_name": "Truck", "label": "货车", "confidence": 0.85, "bbox": [...]}],
  "alert_level": "warning",
  "message": "检测到 1 辆货车"
}
```

---

## 14. 小汽车检测 🆕

**POST** `/detect/car`

返回字段同电瓶车检测。

```json
{
  "success": true,
  "has_vehicle": true,
  "vehicle_count": 3,
  "detections": [
    {"class_name": "Car", "label": "小汽车", "confidence": 0.91, "bbox": [...]},
    {"class_name": "Car", "label": "小汽车", "confidence": 0.88, "bbox": [...]},
    {"class_name": "Car", "label": "小汽车", "confidence": 0.84, "bbox": [...]}
  ],
  "alert_level": "critical",
  "message": "检测到 3 辆小汽车"
}
```

---

## Python 调用示例

```python
import requests

# 安全帽检测
resp = requests.post("https://gtt666.cn/detect/helmet",
    files={"file": open("workers.jpg", "rb")})
data = resp.json()
print(f"未佩戴安全帽: {data['persons_without_helmet']} 人")

# 睡岗检测
resp = requests.post("https://gtt666.cn/detect/sleeping",
    files={"file": open("guard.jpg", "rb")})
data = resp.json()
print(f"睡岗: {data['sleeping_count']} 人")
```

---

## Web Demo

统一测试页面（支持切换所有检测类型）：

👉 https://gtt666.cn/demo

---

## 错误码

| HTTP 状态码 | 说明 |
|------------|------|
| 200 | 成功 |
| 400 | 不支持的图片格式或无法解析 |
| 413 | 图片过大（超过 20MB） |
| 502 | 服务未启动 |
