# CV Animal Detector - Docker 镜像
# 适用于腾讯云轻量应用服务器 (amd64)

FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖（OpenCV 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY main.py .

# 预下载模型（构建时缓存，避免每次启动下载）
RUN python -c "from ultralytics import YOLO; YOLO('yolov8s-oiv7.pt')"

EXPOSE 8000

# 多 worker 启动，利用多核
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
