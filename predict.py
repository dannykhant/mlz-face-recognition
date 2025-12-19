import json
from io import BytesIO
from urllib import request
from PIL import Image

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

session = ort.InferenceSession(
    "face_recognition_v202510191652.onnx", providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

with open("labels.json", "r") as f_in:
    classes = json.load(f_in)


class PredictRequest(BaseModel):
    url: str = "http://bit.ly/4j4Y0Uo"


class PredictResponse(BaseModel):
    top_class: str
    top_score: float
    predictions: dict


def pytorch_preprocessing(X):
  X = X / 255.

  mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
  std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

  # batch, height, width, channels => batch, channels, height, width
  X = X.transpose(0, 3, 1, 2)
  X = (X - mean) / std

  return X.astype(np.float32)


def download_image(url):
    req = request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )
    with request.urlopen(req) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def preprocess(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    small = img.resize((224, 224), Image.NEAREST) # type: ignore
    x = np.array(small, dtype='float32')
    batch = np.expand_dims(x, axis=0)
    return pytorch_preprocessing(batch)


def predict_single(url: str):
    img = download_image(url)
    X = preprocess(img)

    result = session.run([output_name], {input_name: X})
    preds = result[0][0].tolist() # type: ignore

    return dict(zip(classes, preds))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest) -> PredictResponse:
    preds = predict_single(req.url)
    top_class = max(preds, key=preds.get) # type: ignore
    top_score = preds[top_class]

    return PredictResponse(
        top_class=top_class,
        top_score=top_score,
        predictions=preds
    )
