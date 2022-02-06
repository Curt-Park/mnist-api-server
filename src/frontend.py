"""A simple drawable mnist frontend.

This script is a little modified from:
    https://github.com/rahulsrma26/streamlit-mnist-drawable

- Author: Jinwoo Park
- Email: www.jwpark.co.kr@gmail.com
"""


import cv2
import streamlit as st
import torch
from streamlit_drawable_canvas import st_canvas

from ml.data import get_preprocessor

model = torch.jit.load("mnist_cnn.pt")
model.eval()

# img preprocessor
transform = get_preprocessor()

st.title("Digit Recognizer")
st.markdown("Try to write a digit! (0 ~ 9)")

SIZE = 192
mode = st.checkbox("☑ - freedraw  |  ☐ - transform", True)
CanvasResult = st_canvas(
    fill_color="#000000",
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw" if mode else "transform",
    key="canvas",
)

if CanvasResult.image_data is not None:
    img = cv2.resize(CanvasResult.image_data.astype("uint8"), (28, 28))
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.write("Model Input")
    st.image(rescaled)

if st.button("Predict"):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inp = torch.unsqueeze(transform(image), dim=0)
    prediction = torch.argmax(model(inp)).item()
    st.write(f"result: {prediction}")
