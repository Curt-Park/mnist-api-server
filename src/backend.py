"""A simple inferece server.

This script is a little modified from:
    https://github.com/KiLJ4EdeN/fastapi_tf-keras_example

- Author: Jinwoo Park
- Email: www.jwpark.co.kr@gmail.com
"""
from typing import Any, Dict

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel

from ml.data import get_preprocessor


class PredictApiData(BaseModel):
    """Predict API Data."""

    image: Any


# create a fastapi app instance
app = FastAPI()

# img preprocessor
transform = get_preprocessor()

# load the trained model
model = torch.jit.load("mnist_cnn.pt")
model.eval()


@app.post("/predict")
async def predict(data: PredictApiData) -> Dict[str, str]:
    """Run async prediction function."""
    image = np.array(data.image, dtype=np.float32)
    inp = torch.unsqueeze(transform(image), dim=0)
    prediction = torch.argmax(model(inp)).item()
    return {"label": str(prediction)}
