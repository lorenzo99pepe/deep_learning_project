import streamlit as st
import cv2

st.set_page_config(
    page_title="Brain Tumor Segmentation",
    layout="wide",
    initial_sidebar_state="auto",
    # page_icon = st.image(Image.open(str(icon_path))), #provide st.image() with (ie. a PIL array for example)
)

import torch
import numpy as np
import pandas as pd
from PIL import Image
import os
from pathlib import Path
from matplotlib import pyplot as plt

st.title("Your Brain Tumor Segmentation")


models = []
models_path = Path(os.getcwd()) / "models"
for mod in models_path.iterdir():
    if ".pt" in str(mod):
        models.append(mod.name)
models = pd.Series(models)

model_option = st.selectbox(
    "Select the model",
    models,
    index=0,
    key=None,
    help="Here you will select one of the models available to segment your image",
)

model = torch.load(Path(os.getcwd()) / "models" / model_option)
model.eval()


uploaded_files = st.file_uploader(
    "Upload RGB Brain Tumor images",
    type=["png", "jpeg", "jpg"],
    accept_multiple_files=True,
    key=None,
    help="Drag and drop or browse to select a png or jpeg file that will be segmented",
)


predictions = []
for img in uploaded_files:
    if img is not None:
        img_proc = np.array(Image.open(img))
        img_proc = img_proc[:, :, 0]
        input_tensor = (
            torch.tensor(np.array(img_proc)).expand(3, -1, -1)
            .type(torch.ShortTensor)
            .float()
        )
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(input_batch)["out"][0]
        output_predictions = torch.amax(output, 0).numpy()
        threshold_min = np.percentile(output_predictions, 90)
        threshold_mid = np.percentile(output_predictions, 95)
        threshold_max = np.percentile(output_predictions, 99)

        output_pred = output_predictions
        output_pred = np.where(output_predictions > threshold_min, threshold_min, 0)
        output_pred = np.where(output_predictions > threshold_mid, threshold_mid, output_pred)
        output_pred = np.where(output_predictions > threshold_max, threshold_max, output_pred)

        predictions.append(np.array(output_pred))

predictions = np.array(predictions)

if predictions != []:
    for i in range(len(predictions)):
        st.image([uploaded_files[i], predictions[i]], clamp=True, channels="RGB")
