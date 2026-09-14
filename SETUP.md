# CLIP Pet Classifier — Setup Guide

Tài liệu hướng dẫn cài đặt và khởi chạy dự án tại local.

## Yêu cầu hệ thống
- **Python** >= 3.10
- **Node.js** >= 18
- **Docker Desktop** (chạy PostgreSQL)

---

## 1. Khởi động Database

Khởi chạy PostgreSQL qua Docker:

```bash
docker compose up -d
```

---

## 2. Cài đặt & Chạy Backend (FastAPI)

```bash
cd backend

# Khởi tạo và kích hoạt virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
# source venv/bin/activate    # macOS/Linux

# Cài đặt thư viện
pip install -r requirements.txt

# Cấu hình biến môi trường
cp .env.example .env

# Chạy DB migrations và seed dữ liệu ban đầu
alembic upgrade head
cd .. && python scripts/seed_breeds.py
cd backend

# Khởi chạy server
uvicorn app.main:app --reload --port 8000
```

---

## 3. Cài đặt & Chạy Frontend (React Vite)

Mở một terminal mới:

```bash
cd frontend

# Cài đặt dependencies
npm install

# Khởi chạy ứng dụng
npm run dev
```
