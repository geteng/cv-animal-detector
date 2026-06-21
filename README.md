# CV Animal Detector - 部署指南

## 1. 在腾讯云轻量应用服务器上部署

### 前置条件
- 腾讯云轻量应用服务器（建议 2 核 4GB 以上）
- 系统：Ubuntu 22.04 / CentOS 7+ 或 腾讯云提供的 Docker 镜像
- 已安装 Docker（腾讯云应用镜像通常已预装）

### 部署步骤

```bash
# 1. 将项目上传到服务器
scp -r cv-animal-detector root@<你的服务器IP>:/opt/

# 2. SSH 登录服务器
ssh root@<你的服务器IP>

# 3. 构建 Docker 镜像
cd /opt/cv-animal-detector
docker build -t cv-animal-detector .

# 4. 启动容器（使用 host 网络或端口映射）
docker run -d \
  --name cv-detector \
  --restart unless-stopped \
  -p 8000:8000 \
  cv-animal-detector

# 5. 验证服务
curl http://localhost:8000/
curl http://localhost:8000/health
```

### 防火墙设置
在腾讯云控制台 → 轻量应用服务器 → 防火墙，添加规则：
- 端口：8000
- 协议：TCP
- 策略：允许

## 2. API 使用说明

### 单张检测

```bash
curl -X POST http://<服务器IP>:8000/detect \
  -F "file=@test.jpg"
```

返回示例：
```json
{
  "success": true,
  "has_animal": true,
  "detections": [
    {
      "class_name": "Cat",
      "label": "猫",
      "confidence": 0.8921,
      "bbox": [0.12, 0.34, 0.56, 0.78]
    }
  ],
  "count": 1,
  "message": "检测到目标: 猫(0.89)"
}
```

### 批量检测（最多 10 张）

```bash
curl -X POST http://<服务器IP>:8000/detect/batch \
  -F "files=@cat.jpg" \
  -F "files=@dog.jpg"
```

### Python 调用示例

```python
import requests

url = "http://<服务器IP>:8000/detect"
with open("test.jpg", "rb") as f:
    resp = requests.post(url, files={"file": f})
print(resp.json())
```

## 3. 检测类别

| 英文类别    | 中文标签 | 说明           |
|------------|---------|---------------|
| Person     | 人类    | 行人、人物      |
| Snake      | 蛇      | 各类蛇          |
| Lizard     | 蜥蜴    | 蜥蜴、壁虎      |
| Cat        | 猫      | 家猫、野猫      |
| Dog        | 狗      | 家犬、野狗      |
| Wild boar  | 野猪    | 野猪、山猪      |

## 4. 性能参考

- 模型大小：~24MB
- 单张推理：~100-300ms（CPU）/ ~10-30ms（GPU）
- 内存占用：~500MB
- 支持并发请求
