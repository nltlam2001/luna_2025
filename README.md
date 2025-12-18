# Lung Lesion Classification Service

## Giới thiệu

Dịch vụ này cung cấp REST API để dự đoán loại nốt phổi (lung lesion) từ ảnh CT định dạng `.mha`, sử dụng mô hình CNN (ResNet/MedicalNet) kết hợp dữ liệu metadata lâm sàng. Service được xây dựng bằng FastAPI, hỗ trợ xác thực bằng Bearer Token, và có thể chạy trên GPU hoặc CPU.

---

## Cấu trúc thư mục

project/
│
├── api/
│ ├── main.py # FastAPI entrypoint
│ ├── routes.py # Routes predict/health
│ ├── security.py # Bearer token validation
│ └── utils/
│ ├── preprocess.py # Tiền xử lý MHA + metadata
│ └── postprocess.py # Xử lý đầu ra
│
├── model/
│ ├── mednet_resnet.py # Kiến trúc mô hình
│ ├── model_loader.py # Load checkpoint
│ └── weights/
│ └── resnet_*.pth # File trọng số mô hình
│
├── requirements.txt
├── Dockerfile
├── test_data/
│ └── *.mha # File CT mẫu để test
│
└── README.md

yaml
Copy code

---

## Kiến trúc mô hình

### Backbone
- MedicalNet ResNet (18, 50, 101, 200)
- Hỗ trợ đầu vào 3D hoặc 2D slice-based

### Metadata branch
Bao gồm các trường lâm sàng:
- ageAtStudyDate  
- gender  
- seriesInstanceUID  
- coordX, coordY, coordZ  

Các feature metadata được đưa qua MLP và sau đó ghép (concat) với feature ảnh trước lớp phân loại.

### Đầu ra
- probability  
- logit  
- lesionID  
- message  

---

## Hướng dẫn cài đặt và chạy

### 1. Clone repository

```bash
git clone https://github.com/your_repo_url/lesion-service.git
cd lesion-service
```

2. Tạo virtual environment

```bash
python3 -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```

Cài dependencies:
```bash
pip install -r requirements.txt
```

3. Đặt file checkpoint mô hình
Đặt file .pth vào:
```bash
model/weights/
```

Ví dụ:
```bash
model/weights/resnet_200.pth
```

4. Chạy service
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Service sẽ chạy tại:
```bash
http://localhost:8000
```

5. Kiểm tra service
```bash
curl http://localhost:8000/health
```

Gửi request dự đoán bằng curl
Sử dụng file .mha trong thư mục test_data:
```bash
curl -X POST "http://localhost:8000/api/v1/predict/lesion" \
-H "Authorization: Bearer cac-van-de-hien-dai-khmt" \
-F "file=@test_data/1.2.840.113654.2.55.453182006897079545793151028484530532.mha" \
-F "seriesInstanceUID=1.2.840.113654.2.55.453182006897079545793151028484530532" \
-F "patientID=102451" \
-F "studyDate=19990102" \
-F "lesionID=1" \
-F "coordX=-80.96" \
-F "coordY=-131.78" \
-F "coordZ=-92.73" \
-F "ageAtStudyDate=60" \
-F "gender=Male"
```
Ví dụ kết quả trả về
json
Copy code
```bash
{
  "status":"success",
  "data":{
    "seriesInstanceUID":"1.2.840.113654.2.55. 453182006897079545793151028484530532",
    "lesionID":1,
    "probability":0.0324,
    "predictionLabel":0,
    "processingTimeMs":694
  }
}
```

Authentication
Service yêu cầu Bearer Token trong header:
Authorization: Bearer cac-van-de-hien-dai-khmt
Nếu token không hợp lệ, service trả về 401 Unauthorized.

Chạy bằng Docker (tùy chọn)
```bash
docker build -t lesion-service .
docker run -p 8000:8000 lesion-service
```