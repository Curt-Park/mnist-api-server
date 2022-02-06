"""A simple inferece server."""
import io
from typing import Dict

import torch
from fastapi import FastAPI, UploadFile
from PIL import Image

from ml.data import get_preprocessor

# create a fastapi app instance
app = FastAPI()

# img preprocessor
transform = get_preprocessor()

# load the trained model
model = torch.jit.load("mnist_cnn.pt")
model.eval()


@app.post("/predict")
async def predict(data: UploadFile) -> Dict[str, str]:
    """Run async prediction function."""
    img_bytes = io.BytesIO(await data.read())
    image = Image.open(img_bytes)
    inp = torch.unsqueeze(transform(image), dim=0)
    prediction = torch.argmax(model(inp)).item()
    return {"label": str(prediction)}
