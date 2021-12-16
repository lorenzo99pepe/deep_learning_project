# Brain Tumor Semantic Segmentation
Deep learning course project - Group 4: Lorenzo Pepe, Ajay Srivatsa, Doruk Yasa, Ekaterina Zhiganova, Elias Lindbergs

## The Objective
Segmenting brain tumor MRI images to detect their location and sub-regions which are the "enhancing tumor" (ET), the "tumor core" (TC) & the "whole tumor" (WT)

## Approach and Data
We utilized the DeepLab_resnet and DeepLab_mobile architectures. These models were not applied previously to the brain tumor segmentation problem. We downloaded BraTS 2018, 2019 & 2020 challenge datasets for a total of 982 brain cases, 150 images for each brain and different scan times (but we took only one per brain), and 4 different MRI images for each brain: T1, T1Gd, T2, T2-Flair & the segmented ground truth. 

## Repository organization and usage
In order to use correctly the repository follow the next steps. Some minor changes (input data paths, etc.) might be necessary. First of all, do a <code>git clone</code> using the HTTP link. Then, click on this [word](https://www.kaggle.com/andrewmvd/brain-tumor-segmentation-in-mri-brats-2015?select=MICCAI_BraTS2020_TrainingData) to reach the Kaggle Brain tumor segmentation dataset page, scroll down and download the data. 
You can either download all three major directories or not, our code will work anyway, but you will need to change some things (for example, you won't be able to train a model with 2018 data if you don't download them). 
