import streamlit as st

st.set_page_config(
    page_title='Brain Tumor Segmentation',
    layout = 'wide',
    initial_sidebar_state = 'auto',
    #page_icon = st.image(Image.open(str(icon_path))), #provide st.image() with (ie. a PIL array for example)
)

import torch
import numpy as np
import pandas as pd
from PIL import Image
import os
from pathlib import Path
from matplotlib import pyplot as plt

st.title('Your Brain Tumor Segmentation')


models = []
models_path = Path(os.getcwd()) / 'models'
for mod in models_path.iterdir():
    if '.pt' in str(mod):
        models.append(mod.name)
models = pd.Series(models)

model_option = st.selectbox(
    "Select the model", 
    models, 
    index=0,  
    key=None, 
    help='Here you will select one of the models available to segment your image')

model = torch.load(Path(os.getcwd()) / 'models' / model_option)
model.eval()


uploaded_files = st.file_uploader(
    "Upload RGB Brain Tumor images", 
    type=['png', 'jpeg', 'jpg'], 
    accept_multiple_files=True,
    key=None, 
    help="Drag and drop or browse to select a png or jpeg file that will be segmented")


predictions = []
for img in uploaded_files:
    if img is not None:
        img_proc = np.array(Image.open(img))
        img_proc = img_proc[:, :, 0]
        input_tensor = torch.tensor(np.array(img_proc)).expand(3, -1, -1).type(torch.ShortTensor).float()
        input_batch = input_tensor.unsqueeze(0) 

        with torch.no_grad():
            output = model(input_batch)['out'][0]
        output_predictions = output 
        predictions.append(np.array(output_predictions[0]))

predictions = np.array(predictions)

if predictions != []:
    for i in range(len(predictions)):
        st.image([uploaded_files[i], predictions[i]], clamp=True)