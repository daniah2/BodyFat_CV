from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import torch
import cv2
from PIL import Image
import io
import torch.nn as nn
import timm

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

#down.. model using state_dict 
MODEL_PATH = "best_model.pth"  # save state_dict 
model = ImprovedBodyFatModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

IMG_SIZE = 224


# FastAPI
app = FastAPI(title="Body Fat & Muscle Prediction IMG API")


#img proccessing 
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
    img = np.array(image)
    # GRAY → BGR (3 channels)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # Normalize
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    # HWC → CHW (transpose)
    img = np.transpose(img, (2, 0, 1))
    # To tensor
    img = torch.tensor(img).unsqueeze(0).float().to(DEVICE)
    return img


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    # Preprocess image
    img_tensor = preprocess_image(image_bytes)
    # Model prediction
    with torch.no_grad():
        pred = model(img_tensor).cpu().numpy().squeeze()
    # Convert output to %
    bodyfat = round(float(pred[0]) * 100, 2)
    muscle = round(100 - bodyfat, 2)

    return JSONResponse({
        "body_fat_percentage": bodyfat,
        "muscle_percentage": muscle
    })
