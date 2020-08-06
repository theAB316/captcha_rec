import numpy as np
import torch

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile

import config
from model import CaptchaModel
from train import decode_predictions

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# class Item(BaseModel):
#     image: bytes

@app.get("/")
def read_main():
    return {"welcome": "welcome"}

@app.post("/")
def create_upload_file(file: UploadFile = File(...)):
    return {"filename": file.filename}