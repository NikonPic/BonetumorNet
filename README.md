# BonetumorNet
>A Multitask Deep Learning Model for Simultaneous Detection, Segmentation and Classification of Bone Tumors on Radiographs

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NikonPic/bonetumorseg/master?urlpath=voila%2Frender%2F01_segmenter.ipynb) <img src="https://www.code-inspector.com/project/17089/status/svg?branch=master&kill_cache=1" /> <img src="https://www.code-inspector.com/project/17089/score/svg?branch=master&kill_cache=1" />


 <img src="results\demo.gif" alt="Drawing" style="width: 500px;">


## Setup

* Install the apt.txt file
* Install the requirements.txt file
* Add all images and their labels to this repository


## What does each file do? 

    .     
    ├── PNG                              # Folder for all raw images in 'png' format
    ├── SEG                              # Folder for all segmentations in 'nrrd' format     
    ├── src                     
    │   ├── categories.py                # Defines all bone tumor categories to be used for evaluation 
    │   ├── detec_helper.py              # Contains functions for evaluation of the model
    │   ├── eval_doctors.py              # Script to evaluate the results of the doctors
    │   ├── utils_detectron.py           # Utilities for training the model and augmentations
    │   ├── utils_tumor.py               # Utilities for preparing the dataset for training
    │   └── intrareader_reliability.py   # Evaluation of multiple readers on the segmentation performance
    ├── datainfo.csv                     # Contains the labels and filenames
    ├── main_notebook.ipynb              # The main runner script to train and evaluate the model.
    └── requirements.txt                 # Dependencies

# Citation

If you use this project in any of your work, please cite:

```
tbd.
```