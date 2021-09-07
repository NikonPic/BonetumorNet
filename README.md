# BonetumorNet
>A Multitask Deep Learning Model for Simultaneous Detection, Segmentation and Classification of Bone Tumors on Radiographs developed by Nikolas Wilhelm.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NikonPic/bonetumorseg/master?urlpath=voila%2Frender%2F01_segmenter.ipynb) <img src="https://www.code-inspector.com/project/17089/status/svg?branch=master&kill_cache=1" /> <img src="https://www.code-inspector.com/project/17089/score/svg?branch=master&kill_cache=1" />


 <img src="results\demo.gif" alt="Drawing" style="width: 500px;">


## Setup

* Install the apt.txt file.
* Install the requirements.txt file.
* Add all images and their labels to this repository.


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
@article{doi:10.1148/radiol.2021204531,
  author   = {von Schacky, Claudio E. and Wilhelm, Nikolas J. and Schäfer, Valerie S. and Leonhardt, Yannik and Gassert, Felix G. and Foreman, Sarah C. and Gassert, Florian T. and Jung, Matthias and Jungmann, Pia M. and Russe, Maximilian F. and Mogler, Carolin and Knebel, Carolin and von Eisenhart-Rothe, Rüdiger and Makowski, Marcus R. and Woertler, Klaus and Burgkart, Rainer and Gersing, Alexandra S.},
  title    = {Multitask Deep Learning for Segmentation and Classification of Primary Bone Tumors on Radiographs},
  journal  = {Radiology},
  volume   = {0},
  number   = {0},
  pages    = {204531},
  year     = {2021},
  doi      = {10.1148/radiol.2021204531},
  url      = { 
        https://doi.org/10.1148/radiol.2021204531
},
  eprint   = { 
        https://doi.org/10.1148/radiol.2021204531
},
}
```