import numpy as np
import torch

from fastapi import FastAPI, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import config
from model import CaptchaModel
from train import decode_predictions

# Create FastAPI app
app = FastAPI()

# Enable CORS
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_main():
    return {"welcome": "welcome"}

@app.post("/predict")
def create_upload_file(file: bytes = File(...)):
    return {"file size": len(file)}