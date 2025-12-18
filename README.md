# Lung Lesion Classification Service

## Giới thiệu

Dịch vụ này cung cấp REST API để dự đoán loại nốt phổi từ ảnh CT định dạng `.mha`, sử dụng mô hình CNN ConVNeXt kết hợp dữ liệu metadata lâm sàng. Service được xây dựng bằng FastAPI, hỗ trợ xác thực bằng Bearer Token, và có thể chạy trên CPU.

---

## Cấu trúc thư mục
```text
project/
│
├── middlewares/
│ └── auth.py
│
├── utils/
│ ├── error_codes.py 
│ ├── io_utils.py
│ ├── logger.py
│ ├── preprocess.py
│ └── timer.py
│
├── models/
│ ├── model.py
│ └── best_model.pth
│
├── requirements.txt
├── Dockerfile
├── test_data/
│ └── *.mha
│
├── app.py
│
└── README.md
```

---

## Kiến trúc mô hình

### Backbone
- ConVNeXt
- Hỗ trợ đầu vào 3D hoặc 2D slice-based

### Metadata branch
Bao gồm các trường:
- ageAtStudyDate  
- gender  

Các feature metadata được đưa qua MLP và sau đó ghép với feature ảnh trước lớp phân loại.

---

## Hướng dẫn cài đặt và chạy

### 1. Clone repository

```bash
git clone https://github.com/nltlam2001/luna_2025.git
cd luna_2025
```

### 2. Build Docker Image

```bash
docker build -t luna_api:v1.0.0 .
```

Nếu gặp lỗi ```Temporary failure resolving 'deb.debian.org'``` thì thử:

```bash
docker build --network=host -t luna_api:v1.0.0 .
```

### 3. Run Container
```bash
docker run --rm -p 8000:8000 \
  --name luna_api \
  luna_api:v1.0.0
```

### 4. Test API bằng curl
#### 4.1 Health Check
```bash
curl http://localhost:8000/health
```

#### 4.2 Predict (upload file .mha, multipart/form-data)
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

Nếu bạn chạy curl ở ngoài thư mục project, hãy dùng đường dẫn tuyệt đối cho file:

```bash
-F "file=@/var/account/user/test_data/sample.mha"
```

Ví dụ kết quả trả về
```json
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

#### 4.3 Authentication
Service yêu cầu Bearer Token trong header:
```bash
Authorization: Bearer cac-van-de-hien-dai-khmt
```

Nếu token không hợp lệ, service trả về ```401 Unauthorized.```