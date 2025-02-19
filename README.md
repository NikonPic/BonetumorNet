# BonetumorNet
>Multitask Deep Learning for Segmentation and Classification of Primary Bone Tumors on Radiographs -> developed by Nikolas Wilhelm.




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
@article{vonSchacky2021,
  doi = {10.1148/radiol.2021204531},
  url = {https://doi.org/10.1148/radiol.2021204531},
  year = {2021},
  month = sep,
  publisher = {Radiological Society of North America ({RSNA})},
  pages = {204531},
  author = {Claudio E. von Schacky and Nikolas J. Wilhelm and Valerie S. Sch\"{a}fer and Yannik Leonhardt and Felix G. Gassert and Sarah C. Foreman and Florian T. Gassert and Matthias Jung and Pia M. Jungmann and Maximilian F. Russe and Carolin Mogler and Carolin Knebel and R\"{u}diger von Eisenhart-Rothe and Marcus R. Makowski and Klaus Woertler and Rainer Burgkart and Alexandra S. Gersing},
  title = {Multitask Deep Learning for Segmentation and Classification of Primary Bone Tumors on Radiographs},
  journal = {Radiology}
}
```
