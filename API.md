# 🐾 CV Animal Detector API 文档

> 模型：YOLOv8s Open Images V7  
> 检测类别：人类、蛇、蜥蜴、猫、狗、野猪  
> 更新时间：2026-06-22

---

## 服务地址

| 方式 | 地址 |
|------|------|
| HTTPS（推荐） | https://gtt666.cn |
| HTTPS（IP） | https://106.54.38.207 |
| HTTP（自动跳转 HTTPS） | http://gtt666.cn |

---

## 1. 服务信息

**GET** `/`

```bash
curl https://gtt666.cn/
```

返回：

```json
{
  "service": "CV Animal Detector",
  "version": "1.0.0",
  "status": "running"
}
```

---

## 2. 健康检查

**GET** `/health`

```bash
curl https://gtt666.cn/health
```

返回：

```json
{
  "status": "healthy",
  "model": "yolov8s-oiv7.pt"
}
```

---

## 3. 动物 + 人类检测（通用）

**POST** `/detect`

上传一张图片，检测是否包含人类、爬行动物（蛇、蜥蜴）、小型兽类（猫、狗、野猪）。

### 请求

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | File | 是 | 图片文件，支持 JPG / PNG / BMP / WEBP |

### cURL 示例

```bash
curl -X POST https://gtt666.cn/detect -F "file=@image.jpg"
```

### Python 示例

```python
import requests

resp = requests.post("https://gtt666.cn/detect",
    files={"file": open("cat.jpg", "rb")})
print(resp.json())
```

### 返回字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `success` | bool | 请求是否成功 |
| `has_animal` | bool | 是否检测到目标 |
| `detections` | array | 检测结果列表 |
| `detections[].class_name` | string | 英文类别名 |
| `detections[].label` | string | 中文标签 |
| `detections[].confidence` | float | 置信度（0~1） |
| `detections[].bbox` | array | 归一化边界框 [x1, y1, x2, y2] |
| `count` | int | 检测到的目标数量 |
| `message` | string | 结果描述 |

### 返回示例

**检测到目标：**

```json
{
  "success": true,
  "has_animal": true,
  "detections": [
    {
      "class_name": "Cat",
      "label": "猫",
      "confidence": 0.8921,
      "bbox": [0.1234, 0.3456, 0.5678, 0.7890]
    }
  ],
  "count": 1,
  "message": "检测到目标: 猫(0.89)"
}
```

**未检测到：**

```json
{
  "success": true,
  "has_animal": false,
  "detections": [],
  "count": 0,
  "message": "未检测到目标动物或人类"
}
```

---

## 4. 人员入侵检测（周界围挡专用） 🆕

**POST** `/detect/person`

专门用于周界围挡等场景的人员入侵识别。只检测 Person 类别，过滤极小误检目标，返回告警级别。

### 请求

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | File | 是 | 图片文件，支持 JPG / PNG / BMP / WEBP |

### cURL 示例

```bash
curl -X POST https://gtt666.cn/detect/person -F "file=@scene.jpg"
```

### Python 示例

```python
import requests

resp = requests.post("https://gtt666.cn/detect/person",
    files={"file": open("scene.jpg", "rb")})

data = resp.json()
if data["has_person"]:
    print(f"⚠️ 检测到 {data['person_count']} 名人员，告警级别: {data['alert_level']}")
else:
    print("✅ 安全，未检测到人员")
```

### 返回字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `success` | bool | 请求是否成功 |
| `has_person` | bool | 是否检测到人员 |
| `person_count` | int | 检测到的人员数量 |
| `detections` | array | 人员检测结果列表 |
| `detections[].class_name` | string | 固定为 "Person" |
| `detections[].label` | string | 固定为 "人员" |
| `detections[].confidence` | float | 置信度（0~1） |
| `detections[].bbox` | array | 归一化边界框 [x1, y1, x2, y2] |
| `alert_level` | string | 告警级别：`none` / `warning` / `critical` |
| `message` | string | 结果描述 |

### 告警级别说明

| 级别 | 含义 | 条件 |
|------|------|------|
| `none` | 安全 | 未检测到人员 |
| `warning` | 警告 | 检测到 1 名人员 |
| `critical` | 危险 | 检测到 2 名及以上人员 |

### 返回示例

**无人：**

```json
{
  "success": true,
  "has_person": false,
  "person_count": 0,
  "detections": [],
  "alert_level": "none",
  "message": "未检测到人员"
}
```

**1 人：**

```json
{
  "success": true,
  "has_person": true,
  "person_count": 1,
  "detections": [
    {
      "class_name": "Person",
      "label": "人员",
      "confidence": 0.7452,
      "bbox": [0.3625, 0.24, 0.6441, 0.9173]
    }
  ],
  "alert_level": "warning",
  "message": "检测到 1 名人员 (置信度: 0.75)"
}
```

**多人：**

```json
{
  "success": true,
  "has_person": true,
  "person_count": 3,
  "detections": [
    {"class_name": "Person", "label": "人员", "confidence": 0.91, "bbox": [0.1, 0.2, 0.3, 0.8]},
    {"class_name": "Person", "label": "人员", "confidence": 0.85, "bbox": [0.5, 0.1, 0.7, 0.7]},
    {"class_name": "Person", "label": "人员", "confidence": 0.72, "bbox": [0.7, 0.3, 0.9, 0.9]}
  ],
  "alert_level": "critical",
  "message": "检测到 3 名人员 (置信度: 0.91, 0.85, 0.72)"
}
```

---

## 5. 批量检测

**POST** `/detect/batch`

一次上传多张图片（最多 10 张），返回每张的检测结果。

```bash
curl -X POST https://gtt666.cn/detect/batch \
  -F "files=@cat.jpg" \
  -F "files=@dog.jpg"
```

---

## 6. Web 测试页面

| 页面 | 地址 |
|------|------|
| 动物检测 Demo | https://gtt666.cn/demo |
| 人员入侵检测 Demo 🆕 | https://gtt666.cn/demo/person |

浏览器打开即可拖拽图片测试，检测结果会绘制边界框。

---

## 7. 检测类别对照

| 英文类别 | 中文标签 | 接口 |
|---------|---------|------|
| Person | 人类 / 人员 | `/detect` + `/detect/person` |
| Snake | 蛇 | `/detect` |
| Lizard | 蜥蜴 | `/detect` |
| Cat | 猫 | `/detect` |
| Dog | 狗 | `/detect` |
| Wild boar | 野猪 | `/detect` |

---

## 8. 错误码

| HTTP 状态码 | 说明 |
|------------|------|
| 200 | 成功 |
| 400 | 不支持的图片格式或无法解析 |
| 413 | 图片过大（超过 20MB） |
| 502 | 服务未启动 |
