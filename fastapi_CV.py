from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import torch
import cv2
from PIL import Image
import io
import torch.nn as nn
import timm
from fastapi.middleware.cors import CORSMiddleware

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define model
class ImprovedBodyFatModel(nn.Module):
    def __init__(self, model_name='efficientnet_b2', pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        n_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# load model
MODEL_PATH = "best_model.pth"
model = ImprovedBodyFatModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

IMG_SIZE = 224

# FastAPI
app = FastAPI(title="Body Fat & Muscle Prediction IMG API")

# CORS middleware للسماح للصفحة بالوصول للـ API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ضع هنا رابط موقعك إذا تريد تقييد الوصول
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# image preprocessing
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img).unsqueeze(0).float().to(DEVICE)
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img_tensor = preprocess_image(image_bytes)
        with torch.no_grad():
            pred = model(img_tensor).cpu().numpy().squeeze()
        bodyfat = round(float(pred[0]) * 100, 2)
        muscle = round(100 - bodyfat, 2)
        return JSONResponse({
            "body_fat_percentage": bodyfat,
            "muscle_percentage": muscle
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
