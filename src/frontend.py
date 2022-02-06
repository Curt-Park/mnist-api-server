"""A simple drawable mnist frontend.

This script is a little modified from:
    https://github.com/rahulsrma26/streamlit-mnist-drawable

- Author: Jinwoo Park
- Email: www.jwpark.co.kr@gmail.com
"""


import json
import os
import urllib

import cv2
import requests
import streamlit as st
from streamlit_drawable_canvas import st_canvas

if os.environ.get("BACKEND_URL") is not None:
    BACKEND_URL = str(os.environ.get("BACKEND_URL"))
else:
    BACKEND_URL = "http://localhost:8000"
PREDICT_URL = urllib.parse.urljoin(BACKEND_URL, "predict")


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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    try:
        response_predict = requests.post(
            url=PREDICT_URL,
            data=json.dumps({"input_image": img.tolist()}),
        )

        if response_predict.ok:
            res = response_predict.json()
            st.write(f"Prediction: {res['label']}")
        else:
            st.write("Some error occured")

    except ConnectionError:
        st.write("Couldn't reach backend")
