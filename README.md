# BonetumorNet
A Multitask Deep Learning Model for Simultaneous Detection, Segmentation and Classification of Bone Tumors on Radiographs

<img src="results\segmentation.PNG" alt="Drawing" style="width: 500px;">

## Setup

* Install the apt.txt file
* Install the requirements.txt file
* Add all images and their labels to this repository

## This project contains:

1. PNG2: Folder withh all png files of the tumors to analyse
2. SEG: Folder with all Segmented files: -> Picture file (.png) and segmented file (.nrrd)
3. datainfo.csv -> Contains all relevant relations

## Test the model:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NikonPic/bonetumorseg/master?urlpath=voila%2Frender%2F01_segmenter.ipynb)

## What does each file do? 

    .                  
    ├── src                     
    │   ├── categories.py                # Defines all bone tumor categories to be used for evaluation 
    │   ├── detec_helper.py              # Contains functions for evaluation of the model
    │   ├── eval_doctors.py              # Script to evaluate the results of the doctors
    │   ├── utils_detectron.py           # Utilities for training the model and augmentations
    │   ├── utils_tumor.py               # Utilities for preparing the dataset for training
    │   └── intrareader_reliability.py   # Evaluation of multiple readers on the segmentation performance
    ├── main.py                          # The main runner script to launch jobs.
    └── requirements.txt                 # Dependencies

# Citation

If you use this project in any of your work, please cite:

```
tbd.
```