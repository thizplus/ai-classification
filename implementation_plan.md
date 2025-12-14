# Phase 1 Implementation Plan - AI Classification Service

## Overview

วางแผน implement ระบบ AI Classification ที่เชื่อมต่อกับ GoFiber backend และ ESP32-CAM

---

## ESP32-CAM Current State (จากการวิเคราะห์ code)

### Project Location
- **ESP32-CAM**: `C:\Users\Admin\Documents\PlatformIO\Projects\smart-trash-picker`
- **GoFiber**: `D:\Admin\Desktop\MY PROJECT\__serkk\Smart Trash\gofiber-smart-trash`

### Current Flow
```
ESP32-CAM
    │
    ├─ 1. Capture photo (OV2640, 640x480 JPEG)
    ├─ 2. GET /api/upload-url → ได้ presigned URL
    ├─ 3. PUT {upload_url} → Upload to R2
    └─ 4. POST /api/trash → Save record
            │
            ▼
       Response (ไม่มี classification)
            │
            ▼
       LED Pattern (success/error เท่านั้น)
```

### Hardware ที่มี
- **Camera**: OV2640 (640x480 JPEG)
- **LED_STATUS**: GPIO 33 (active low) - แสดง success/error
- **LED_FLASH**: GPIO 4 (ยังไม่ได้ใช้)
- **Serial**: 115200 baud (debug)
- **Web UI**: Port 80 (preview + capture button)

### ปัญหาที่ต้องแก้
**ESP32 ต้องรู้ว่าขยะเป็นอะไร** → ต้องทำ **Sync** (GoFiber รอ AI ก่อน return)

---

## Target Flow (เป้าหมาย)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NEW SYNC FLOW                                     │
│                                                                             │
│  ESP32-CAM                GoFiber              Python AI          R2       │
│      │                       │                     │               │        │
│      │  1. GET /upload-url   │                     │               │        │
│      │─────────────────────▶│                     │               │        │
│      │◀─────────────────────│                     │               │        │
│      │                       │                     │               │        │
│      │  2. PUT image ────────────────────────────────────────────▶│        │
│      │◀──────────────────────────────────────────────────────────│        │
│      │                       │                     │               │        │
│      │  3. POST /api/trash   │                     │               │        │
│      │─────────────────────▶│                     │               │        │
│      │                       │  4. POST /classify  │               │        │
│      │                       │───────────────────▶│               │        │
│      │                       │                     │  5. Fetch     │        │
│      │                       │                     │─────────────▶│        │
│      │                       │                     │◀─────────────│        │
│      │                       │                     │  6. Classify  │        │
│      │                       │◀───────────────────│               │        │
│      │                       │  7. Save + Return   │               │        │
│      │◀─────────────────────│                     │               │        │
│      │                       │                     │               │        │
│      │  8. แสดงผล (LED/Serial/LCD)                                         │
│      ▼                                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### New Response Format

```json
// POST /api/trash Response (ใหม่ - มี classification)
{
  "success": true,
  "data": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "device_id": "ESP32CAM001",
    "image_url": "https://pub-xxx.r2.dev/trash/.../xxx.jpg",
    "latitude": 13.756331,
    "longitude": 100.501762,

    // NEW: Classification result
    "category": "plastic",
    "sub_category": null,
    "confidence": 0.9523,
    "bin_number": 5,
    "bin_label": "พลาสติก",
    "message": "ทิ้งช่อง 5 - พลาสติก",

    "created_at": "2025-12-13T12:00:00Z"
  }
}
```

---

## ESP32-CAM Changes Required

### 1. Update `api.h` - Add Classification Response Struct

```cpp
// api.h (เพิ่ม)

struct ClassificationResult {
  bool hasClassification;
  String category;      // plastic, metal, glass, paper, cardboard, trash
  String subCategory;   // PET, HDPE, PP, PE (nullable)
  float confidence;     // 0.0 - 1.0
  int binNumber;        // 1-6
  String binLabel;      // "พลาสติก", "โลหะ", etc.
  String message;       // "ทิ้งช่อง 5 - พลาสติก"
};

struct TrashRecordResponse {
  bool success;
  String id;
  String deviceId;
  String imageUrl;
  ClassificationResult classification;
};
```

### 2. Update `api.cpp` - Parse Classification

```cpp
// api.cpp - saveTrashRecord() แก้ไข

TrashRecordResponse saveTrashRecord(const TrashRecord& record) {
  TrashRecordResponse result;
  result.success = false;

  HTTPClient http;
  String url = buildUrl("/api/trash");

  // ... existing JSON build code ...

  int httpCode = http.POST(json);
  String response = http.getString();
  http.end();

  if (httpCode == 200 || httpCode == 201) {
    // Parse response
    JsonDocument doc;
    DeserializationError error = deserializeJson(doc, response);

    if (!error && doc["success"] == true) {
      result.success = true;
      result.id = doc["data"]["id"].as<String>();

      // Parse classification (ใหม่)
      if (doc["data"].containsKey("category")) {
        result.classification.hasClassification = true;
        result.classification.category = doc["data"]["category"].as<String>();
        result.classification.confidence = doc["data"]["confidence"].as<float>();
        result.classification.binNumber = doc["data"]["bin_number"].as<int>();
        result.classification.binLabel = doc["data"]["bin_label"].as<String>();
        result.classification.message = doc["data"]["message"].as<String>();

        if (doc["data"].containsKey("sub_category") && !doc["data"]["sub_category"].isNull()) {
          result.classification.subCategory = doc["data"]["sub_category"].as<String>();
        }
      }
    }
  }

  return result;
}
```

### 3. Update `main.cpp` - Display Result

```cpp
// main.cpp - captureAndUpload() แก้ไข

void captureAndUpload() {
  // ... existing capture code ...

  // Step 4: Save record และรับ classification
  TrashRecord record;
  record.deviceId = DEVICE_ID;
  record.imageUrl = urls.imageUrl;
  record.latitude = 13.756331;
  record.longitude = 100.501762;

  TrashRecordResponse response = Api::saveTrashRecord(record);

  if (!response.success) {
    Serial.println("[ERROR] Save failed!");
    Led::error();
    return;
  }

  // NEW: แสดงผล classification
  if (response.classification.hasClassification) {
    Serial.println("\n========== CLASSIFICATION RESULT ==========");
    Serial.printf("Category: %s\n", response.classification.category.c_str());
    Serial.printf("Confidence: %.2f%%\n", response.classification.confidence * 100);
    Serial.printf("Bin: %d - %s\n",
      response.classification.binNumber,
      response.classification.binLabel.c_str());
    Serial.println(response.classification.message);
    Serial.println("============================================\n");

    // LED pattern ตาม category (optional)
    Led::showBinNumber(response.classification.binNumber);
  }

  Serial.println("[SUCCESS] Capture completed!");
  Led::success();
}
```

### 4. Update `led.cpp` - Add Bin Indicator

```cpp
// led.cpp (เพิ่ม)

void showBinNumber(int binNumber) {
  // แสดงเลขช่องด้วยการกระพริบ LED
  // เช่น bin 3 = กระพริบ 3 ครั้ง

  delay(500);  // Pause ก่อนแสดง

  for (int i = 0; i < binNumber; i++) {
    digitalWrite(LED_STATUS, LOW);   // ON
    delay(200);
    digitalWrite(LED_STATUS, HIGH);  // OFF
    delay(200);
  }

  delay(500);  // Pause หลังแสดง
}
```

### 5. (Optional) Add LCD/OLED Display

ถ้ามี OLED display (เช่น SSD1306 128x64):

```cpp
// display.cpp (ใหม่)

#include <Adafruit_SSD1306.h>

Adafruit_SSD1306 display(128, 64, &Wire, -1);

void showClassification(const ClassificationResult& result) {
  display.clearDisplay();
  display.setTextSize(2);
  display.setTextColor(WHITE);

  // Line 1: Category
  display.setCursor(0, 0);
  display.println(result.category);

  // Line 2: Confidence
  display.setTextSize(1);
  display.setCursor(0, 20);
  display.printf("Confidence: %.0f%%", result.confidence * 100);

  // Line 3: Bin number
  display.setTextSize(2);
  display.setCursor(0, 35);
  display.printf("BIN: %d", result.binNumber);

  // Line 4: Thai label
  display.setTextSize(1);
  display.setCursor(0, 55);
  display.println(result.message);

  display.display();
}
```

---

## Bin Mapping (สำหรับ Response)

```
Category      → Bin  → Label (TH)    → Label (EN)
─────────────────────────────────────────────────
cardboard     → 1    → กระดาษแข็ง     → Cardboard
glass         → 2    → แก้ว          → Glass
metal         → 3    → โลหะ          → Metal
paper         → 4    → กระดาษ        → Paper
plastic       → 5    → พลาสติก       → Plastic
trash         → 6    → ขยะทั่วไป      → General Waste
```

---

## Current State (สิ่งที่มีอยู่แล้ว)

### GoFiber Backend
```
Endpoints:
├── GET  /api/upload-url     → Presigned URL สำหรับ upload รูป
├── POST /api/trash          → สร้าง trash record
├── GET  /api/trash          → List trash records
└── GET  /api/trash/:id      → Get trash by ID

Database (trash_records):
├── id          (UUID)
├── device_id   (VARCHAR)
├── image_url   (TEXT)
├── latitude    (DECIMAL)
├── longitude   (DECIMAL)
├── created_at  (TIMESTAMP)
└── updated_at  (TIMESTAMP)
```

**ปัญหา:** ยังไม่มี classification fields!

---

## Target State (สิ่งที่ต้องการ)

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FULL SYSTEM FLOW                               │
│                                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │ ESP32   │───▶│   R2    │───▶│ GoFiber │───▶│ Python  │───▶│ GoFiber │  │
│  │ Camera  │    │ Storage │    │ Backend │    │   AI    │    │  Save   │  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
│       │              │              │              │              │        │
│       ▼              ▼              ▼              ▼              ▼        │
│   1.ถ่ายรูป     2.เก็บรูป    3.สร้าง record  4.Classify    5.Update      │
│                              + เรียก AI      + Return      classification │
└─────────────────────────────────────────────────────────────────────────────┘
```

### New API Flow

```
ESP32 Flow:
1. GET  /api/upload-url?device_id=xxx     → ได้ presigned URL
2. PUT  {upload_url}                       → Upload รูปไป R2
3. POST /api/trash { image_url, ... }      → สร้าง record
   ↓
   GoFiber เรียก Python AI Service โดยอัตโนมัติ
   ↓
4. Response กลับพร้อม classification result
```

---

## Implementation Tasks

### Part A: Python AI Service (ใหม่ทั้งหมด)

```
ai-classification/
├── src/
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── entities/
│   │   │   ├── __init__.py
│   │   │   └── classification_result.py
│   │   ├── enums/
│   │   │   ├── __init__.py
│   │   │   └── trash_category.py
│   │   └── interfaces/
│   │       ├── __init__.py
│   │       ├── classifier.py
│   │       └── image_processor.py
│   │
│   ├── application/
│   │   ├── __init__.py
│   │   ├── classify_trash.py
│   │   └── routing_service.py
│   │
│   ├── infrastructure/
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── base_model.py
│   │   │   └── trash_net.py
│   │   └── image/
│   │       ├── __init__.py
│   │       └── processor.py
│   │
│   └── presentation/
│       ├── __init__.py
│       └── api/
│           ├── __init__.py
│           ├── main.py          # FastAPI app
│           ├── routes.py
│           └── schemas.py
│
├── requirements.txt
├── Dockerfile
└── .env.example
```

### Part B: GoFiber Backend (แก้ไข)

```
แก้ไขไฟล์:
├── domain/models/trash.go           # เพิ่ม classification fields
├── domain/dto/trash.go              # เพิ่ม classification ใน response
├── domain/services/trash_service.go # เพิ่ม interface method
├── application/services/trash_service_impl.go  # เพิ่ม logic เรียก AI
├── pkg/config/config.go             # เพิ่ม AI service URL config
└── infrastructure/ai/               # (ใหม่) AI client adapter
    └── classifier_client.go
```

---

## Detailed Implementation

### Step 1: Update GoFiber Database Model

```go
// domain/models/trash.go (เพิ่ม fields)

type TrashRecord struct {
    ID        uuid.UUID      `gorm:"type:uuid;primaryKey;default:gen_random_uuid()" json:"id"`
    DeviceID  string         `gorm:"type:varchar(20);not null;index" json:"device_id"`
    ImageURL  string         `gorm:"type:text;not null" json:"image_url"`
    Latitude  float64        `gorm:"type:decimal(10,8);not null" json:"latitude"`
    Longitude float64        `gorm:"type:decimal(11,8);not null" json:"longitude"`

    // NEW: Classification fields
    Category       *string   `gorm:"type:varchar(20);index" json:"category"`         // plastic, metal, glass, paper, cardboard, trash
    SubCategory    *string   `gorm:"type:varchar(20)" json:"sub_category"`           // PET, HDPE, PP, PE (สำหรับ plastic)
    Confidence     *float64  `gorm:"type:decimal(5,4)" json:"confidence"`            // 0.0000 - 1.0000
    ClassifiedAt   *time.Time `json:"classified_at"`
    ClassifyError  *string   `gorm:"type:text" json:"classify_error,omitempty"`      // เก็บ error ถ้า classify ไม่สำเร็จ

    CreatedAt time.Time      `json:"created_at"`
    UpdatedAt time.Time      `json:"updated_at"`
    DeletedAt gorm.DeletedAt `gorm:"index" json:"-"`
}
```

**Migration SQL:**
```sql
ALTER TABLE trash_records
ADD COLUMN category VARCHAR(20),
ADD COLUMN sub_category VARCHAR(20),
ADD COLUMN confidence DECIMAL(5,4),
ADD COLUMN classified_at TIMESTAMP,
ADD COLUMN classify_error TEXT;

CREATE INDEX idx_trash_records_category ON trash_records(category);
```

---

### Step 2: Create Python AI Service

#### 2.1 Main FastAPI App

```python
# src/presentation/api/main.py

from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.presentation.api.routes import router
from src.infrastructure.models.trash_net import TrashNetClassifier

# Global classifier instance
classifier = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    global classifier
    print("Loading Trash-Net model...")
    classifier = TrashNetClassifier()
    print("Model loaded!")
    yield
    # Shutdown
    print("Shutting down...")

app = FastAPI(
    title="Smart Trash AI Classification",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router, prefix="/api")

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": classifier is not None}
```

#### 2.2 API Routes

```python
# src/presentation/api/routes.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Optional
import httpx
from PIL import Image
from io import BytesIO

router = APIRouter()

class ClassifyRequest(BaseModel):
    image_url: HttpUrl
    trash_id: Optional[str] = None  # สำหรับ tracking

class ClassifyResponse(BaseModel):
    success: bool
    category: str           # plastic, metal, glass, paper, cardboard, trash
    sub_category: Optional[str] = None  # PET, HDPE, etc.
    confidence: float
    model_used: str
    trash_id: Optional[str] = None

@router.post("/classify", response_model=ClassifyResponse)
async def classify_trash(request: ClassifyRequest):
    """Classify trash image from URL"""
    try:
        # 1. Fetch image from R2
        async with httpx.AsyncClient() as client:
            response = await client.get(str(request.image_url), timeout=30.0)
            response.raise_for_status()
            image_bytes = response.content

        # 2. Load image
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # 3. Classify
        from src.presentation.api.main import classifier
        result = classifier.classify(image)

        return ClassifyResponse(
            success=True,
            category=result.category.value,
            sub_category=result.sub_result.category.value if result.sub_result else None,
            confidence=result.confidence,
            model_used=result.model_used,
            trash_id=request.trash_id
        )

    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")
```

#### 2.3 Trash-Net Classifier

```python
# src/infrastructure/models/trash_net.py

from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

from src.domain.entities.classification_result import ClassificationResult
from src.domain.enums.trash_category import TrashCategory

class TrashNetClassifier:
    """Trash-Net Classifier using HuggingFace model"""

    INDEX_TO_CATEGORY = [
        TrashCategory.CARDBOARD,  # 0
        TrashCategory.GLASS,      # 1
        TrashCategory.METAL,      # 2
        TrashCategory.PAPER,      # 3
        TrashCategory.PLASTIC,    # 4
        TrashCategory.TRASH,      # 5
    ]

    def __init__(self, threshold: float = 0.85):
        self._threshold = threshold
        self._model_name = "prithivMLmods/Trash-Net"

        print(f"Loading model: {self._model_name}")
        self._model = SiglipForImageClassification.from_pretrained(self._model_name)
        self._processor = AutoImageProcessor.from_pretrained(self._model_name)
        print("Model loaded successfully!")

    def classify(self, image: Image.Image) -> ClassificationResult:
        """Classify image"""
        # Preprocess
        inputs = self._processor(images=image, return_tensors="pt")

        # Inference
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)

        # Get prediction
        idx = probs.argmax().item()
        confidence = probs[0][idx].item()

        return ClassificationResult(
            category=self.INDEX_TO_CATEGORY[idx],
            confidence=confidence,
            model_used="trash_net_v1_hf"
        )

    @property
    def confidence_threshold(self) -> float:
        return self._threshold
```

---

### Step 3: GoFiber AI Client

```go
// infrastructure/ai/classifier_client.go

package ai

import (
    "bytes"
    "encoding/json"
    "fmt"
    "net/http"
    "time"
)

type ClassifierClient struct {
    baseURL    string
    httpClient *http.Client
}

type ClassifyRequest struct {
    ImageURL string `json:"image_url"`
    TrashID  string `json:"trash_id,omitempty"`
}

type ClassifyResponse struct {
    Success     bool    `json:"success"`
    Category    string  `json:"category"`
    SubCategory *string `json:"sub_category"`
    Confidence  float64 `json:"confidence"`
    ModelUsed   string  `json:"model_used"`
    TrashID     string  `json:"trash_id"`
}

func NewClassifierClient(baseURL string) *ClassifierClient {
    return &ClassifierClient{
        baseURL: baseURL,
        httpClient: &http.Client{
            Timeout: 60 * time.Second, // AI inference อาจใช้เวลานาน
        },
    }
}

func (c *ClassifierClient) Classify(imageURL, trashID string) (*ClassifyResponse, error) {
    reqBody := ClassifyRequest{
        ImageURL: imageURL,
        TrashID:  trashID,
    }

    jsonBody, err := json.Marshal(reqBody)
    if err != nil {
        return nil, fmt.Errorf("failed to marshal request: %w", err)
    }

    resp, err := c.httpClient.Post(
        c.baseURL+"/api/classify",
        "application/json",
        bytes.NewBuffer(jsonBody),
    )
    if err != nil {
        return nil, fmt.Errorf("failed to call AI service: %w", err)
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {
        return nil, fmt.Errorf("AI service returned status %d", resp.StatusCode)
    }

    var result ClassifyResponse
    if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
        return nil, fmt.Errorf("failed to decode response: %w", err)
    }

    return &result, nil
}
```

---

### Step 4: Update Trash Service (SYNC - รอ AI ก่อน return)

```go
// application/services/trash_service_impl.go (แก้ไข)

// Bin mapping
var binMapping = map[string]struct {
    Number int
    LabelTH string
    LabelEN string
}{
    "cardboard": {1, "กระดาษแข็ง", "Cardboard"},
    "glass":     {2, "แก้ว", "Glass"},
    "metal":     {3, "โลหะ", "Metal"},
    "paper":     {4, "กระดาษ", "Paper"},
    "plastic":   {5, "พลาสติก", "Plastic"},
    "trash":     {6, "ขยะทั่วไป", "General Waste"},
}

func (s *trashServiceImpl) Create(req *dto.CreateTrashRequest) (*dto.TrashResponse, error) {
    // 1. Create record first
    record := &models.TrashRecord{
        DeviceID:  req.DeviceID,
        ImageURL:  req.ImageURL,
        Latitude:  req.Latitude,
        Longitude: req.Longitude,
    }

    if err := s.repo.Create(record); err != nil {
        return nil, err
    }

    // 2. Call AI service (SYNC - รอผลก่อน return)
    classifyResult, err := s.aiClient.Classify(req.ImageURL, record.ID.String())

    if err != nil {
        // AI failed - log error but still return success (record created)
        errMsg := err.Error()
        record.ClassifyError = &errMsg
        s.repo.Update(record)

        // Return without classification
        return s.toResponse(record), nil
    }

    // 3. Update record with classification
    now := time.Now()
    record.Category = &classifyResult.Category
    record.SubCategory = classifyResult.SubCategory
    record.Confidence = &classifyResult.Confidence
    record.ClassifiedAt = &now
    s.repo.Update(record)

    // 4. Return response พร้อม classification
    return s.toResponseWithClassification(record), nil
}

func (s *trashServiceImpl) toResponseWithClassification(record *models.TrashRecord) *dto.TrashResponse {
    resp := &dto.TrashResponse{
        ID:        record.ID,
        DeviceID:  record.DeviceID,
        ImageURL:  record.ImageURL,
        Latitude:  record.Latitude,
        Longitude: record.Longitude,
        CreatedAt: record.CreatedAt,
    }

    // Add classification if available
    if record.Category != nil {
        resp.Category = record.Category
        resp.SubCategory = record.SubCategory
        resp.Confidence = record.Confidence
        resp.ClassifiedAt = record.ClassifiedAt

        // Add bin info
        if bin, ok := binMapping[*record.Category]; ok {
            resp.BinNumber = &bin.Number
            resp.BinLabel = &bin.LabelTH
            msg := fmt.Sprintf("ทิ้งช่อง %d - %s", bin.Number, bin.LabelTH)
            resp.Message = &msg
        }
    }

    return resp
}
```

**สำคัญ:** ใช้ SYNC เพราะ ESP32 ต้องรู้ผลทันทีเพื่อแสดงผล

---

## API Contract (Final)

### POST /api/trash (Updated Response)

```json
// Request (เหมือนเดิม)
{
  "device_id": "DEVICE001",
  "image_url": "https://pub-xxx.r2.dev/trash/DEVICE001/1702468800000.jpg",
  "latitude": 13.736717,
  "longitude": 100.523186
}

// Response (เพิ่ม classification - หลังจาก AI classify เสร็จ)
{
  "success": true,
  "data": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "device_id": "DEVICE001",
    "image_url": "https://pub-xxx.r2.dev/trash/DEVICE001/1702468800000.jpg",
    "latitude": 13.736717,
    "longitude": 100.523186,
    "category": "plastic",
    "sub_category": null,
    "confidence": 0.9523,
    "classified_at": "2025-12-13T12:00:05Z",
    "created_at": "2025-12-13T12:00:00Z"
  }
}
```

### Python AI Service: POST /api/classify

```json
// Request
{
  "image_url": "https://pub-xxx.r2.dev/trash/DEVICE001/1702468800000.jpg",
  "trash_id": "550e8400-e29b-41d4-a716-446655440000"
}

// Response
{
  "success": true,
  "category": "plastic",
  "sub_category": null,
  "confidence": 0.9523,
  "model_used": "trash_net_v1_hf",
  "trash_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

## Environment Variables

### Python AI Service (.env)
```env
# Server
HOST=0.0.0.0
PORT=8081

# Model
MODEL_NAME=prithivMLmods/Trash-Net
CONFIDENCE_THRESHOLD=0.85

# Optional: GPU
CUDA_VISIBLE_DEVICES=0
```

### GoFiber Backend (.env - เพิ่ม)
```env
# AI Service
AI_SERVICE_URL=http://localhost:8081
AI_SERVICE_TIMEOUT=60
```

---

## Implementation Order

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Python AI Service (สร้างใหม่)                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  □ Setup project structure                                                  │
│  □ Implement domain layer (entities, enums)                                 │
│  □ Implement TrashNetClassifier (HuggingFace)                              │
│  □ Implement FastAPI routes                                                 │
│  □ Test locally with sample images                                          │
│  □ Create Dockerfile                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: GoFiber Backend (แก้ไข)                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  □ Update database model (เพิ่ม classification fields)                      │
│  □ Run migration                                                            │
│  □ Update DTOs (request/response)                                           │
│  □ Create AI client adapter                                                 │
│  □ Update trash service (SYNC mode)                                         │
│  □ Add bin mapping logic                                                    │
│  □ Test API with curl/Postman                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: ESP32-CAM (แก้ไข)                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  □ Update api.h - Add ClassificationResult struct                          │
│  □ Update api.cpp - Parse classification from response                     │
│  □ Update main.cpp - Display result on Serial                              │
│  □ Update led.cpp - Add showBinNumber() function                           │
│  □ Test end-to-end                                                         │
│  □ (Optional) Add OLED display support                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: Integration Test                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  □ ESP32 capture → R2 upload → GoFiber → Python AI → Response              │
│  □ Verify classification result on Serial Monitor                          │
│  □ Verify LED blinks correct bin number                                    │
│  □ Verify database has classification data                                  │
│  □ Test error scenarios (AI down, invalid image, etc.)                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Testing Checklist

### Python AI Service
- [ ] Health check endpoint works (`GET /health`)
- [ ] Model loads successfully on startup
- [ ] Can classify image from URL
- [ ] Returns correct category (cardboard/glass/metal/paper/plastic/trash)
- [ ] Returns confidence score (0.0 - 1.0)
- [ ] Handles invalid image URL gracefully
- [ ] Handles timeout gracefully

### GoFiber Integration
- [ ] New fields added to database (category, sub_category, confidence, etc.)
- [ ] Migration runs successfully
- [ ] AI client connects to Python service
- [ ] **SYNC mode works** - waits for AI result before returning
- [ ] Classification result saved to database
- [ ] Response includes bin_number, bin_label, message
- [ ] Error handling when AI service is down

### ESP32-CAM
- [ ] Can parse new response format
- [ ] Displays category on Serial Monitor
- [ ] Displays confidence percentage
- [ ] Displays bin number and label
- [ ] LED blinks correct number of times for bin
- [ ] Handles response without classification (AI error)

### End-to-End Flow
- [ ] ESP32 captures image
- [ ] Image uploads to R2 successfully
- [ ] POST /api/trash returns classification
- [ ] Serial Monitor shows: `Category: plastic, Bin: 5 - พลาสติก`
- [ ] LED blinks 5 times (for plastic)
- [ ] Database record has all classification fields

---

## Deployment Options

### Option A: Same Server (Simple)
```
┌─────────────────────────────────────────┐
│            Single Server                │
│  ┌─────────────┐  ┌─────────────────┐  │
│  │   GoFiber   │  │   Python AI     │  │
│  │   :8080     │─▶│   :8081         │  │
│  └─────────────┘  └─────────────────┘  │
└─────────────────────────────────────────┘
```

### Option B: Docker Compose
```yaml
version: '3.8'
services:
  gofiber:
    build: ./gofiber-smart-trash
    ports:
      - "8080:8080"
    environment:
      - AI_SERVICE_URL=http://ai:8081
    depends_on:
      - ai
      - postgres

  ai:
    build: ./ai-classification
    ports:
      - "8081:8081"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: smartpicker
      POSTGRES_PASSWORD: password
```

---

## Summary - สรุปแผนทั้งหมด

### Components ที่ต้องทำ

| Component | Action | Files |
|-----------|--------|-------|
| **Python AI Service** | สร้างใหม่ | ~10 files |
| **GoFiber Backend** | แก้ไข | ~6 files |
| **ESP32-CAM** | แก้ไข | ~4 files |

### Key Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Sync vs Async | **SYNC** | ESP32 ต้องรู้ผลทันที |
| AI Model | Trash-Net (HuggingFace) | Free, 96% accuracy |
| Framework | FastAPI | Async, fast, easy |
| Communication | HTTP REST | Simple, reliable |

### Expected Response Time

```
Capture → Upload → Classify → Response

With GPU:  ~2-3 seconds total
With CPU:  ~5-7 seconds total (acceptable)
```

### Expected Output (ESP32 Serial Monitor)

```
[MAIN] Starting capture...
[API] GET http://192.168.1.30:8080/api/upload-url?device_id=ESP32CAM001
[API] PUT image (45000 bytes)
[API] POST http://192.168.1.30:8080/api/trash

========== CLASSIFICATION RESULT ==========
Category: plastic
Confidence: 95.23%
Bin: 5 - พลาสติก
ทิ้งช่อง 5 - พลาสติก
============================================

[SUCCESS] Capture completed!
```

---

## Next Steps

เมื่อคุณพร้อม บอกผมได้เลย:

1. **"เริ่ม Step 1"** → ผมสร้าง Python AI Service
2. **"เริ่ม Step 2"** → ผมแก้ GoFiber Backend
3. **"เริ่ม Step 3"** → ผมแก้ ESP32-CAM code
4. **"ทำทั้งหมด"** → ผมทำทุก Step ต่อเนื่อง

---

## Questions Before Start

1. **GPU?**
   - มี GPU สำหรับ inference ไหม?
   - ถ้าไม่มี ใช้ CPU ได้ (ช้ากว่า ~2-3 วินาที แต่ยังใช้งานได้)

2. **Hosting?**
   - จะ deploy Python AI ที่ไหน?
   - **Local PC** (แนะนำตอนพัฒนา): ใช้ RAM ~4GB
   - **Cloud** (Production): Render, Railway, หรือ VPS

3. **ESP32 Display?**
   - มี OLED/LCD display ต่อกับ ESP32 ไหม?
   - ถ้าไม่มี ใช้ Serial Monitor + LED ก็พอสำหรับตอนนี้
