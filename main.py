
# %%
import ipywidgets as widgets
from PIL import Image, ImageOps, ImageDraw, ImageFont

from sklearn.metrics import confusion_matrix
from tqdm.notebook import tqdm
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

import detectron2
from detectron2.utils.visualizer import ColorMode
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

import torch
import torchvision
from src.utils_tumor import get_df_paths, get_cocos_from_df, get_advanced_dis_df, format_seg_names
import src.utils_detectron as ud
import src.detec_helper as dh
from src.utils_detectron import F_KEY, CLASS_KEY, ENTITY_KEY
from src.categories import make_categories_advanced, make_categories, cat_mapping_new, cat_naming_new, reverse_cat_list


# import some common libraries
import numpy as np
import random
import os
import sys
import nrrd
import cv2
import pandas as pd

from sklearn.metrics import cohen_kappa_score

setup_logger()
cfg = get_cfg()
print(detectron2.__version__)

# %% Define the model / Load and train it!

# names
cfg.OUTPUT_DIR = "./models"
model_str = "model_final.pth"

# iters
cfg.SOLVER.MAX_ITER = 40000


# General definitions:
mask_it = True
simple = True
advanced = True
train = False


# load dataframe and paths
df, paths = get_df_paths()
df_ex, paths_ex = get_df_paths(mode=True)

# prepare all datasets to coco format
if train:
    print('remake coco')
    #cocos = get_cocos_from_df(df, paths, save=True, seg=True, simple=simple)

dis = get_advanced_dis_df(df)
df.head()

# Register datasets
path = os.path.join(os.getcwd())
pic_path = os.path.join(path, "PNG2")
pic_path_external = os.path.join(path, "PNG2_external")

register_coco_instances(
    "my_dataset_train", {}, os.path.join(path, "train.json"), pic_path
)
register_coco_instances(
    "my_dataset_valid", {}, os.path.join(path, "test.json"), pic_path
)
register_coco_instances(
    "my_dataset_test", {}, os.path.join(
        path, "test_external.json"), pic_path_external
)

# select the right network
if mask_it:
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
        )
    )
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    )

# no masking required?
else:
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl"

# select datasets
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_valid",)
cfg.TEST.EVAL_PERIOD = 10000
cfg.DATALOADER.NUM_WORKERS = 0

# training parameters
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR

# roi and classes
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512

# set to "RepeatFactorTrainingSampler" in order to allow balanced sampling
cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
cfg.DATALOADER.REPEAT_THRESHOLD = 0.3

if advanced:
    _, cat_list = make_categories_advanced(simple=simple)
else:
    _, cat_list = make_categories(simple=simple)

num_classes = len(list(cat_list)) if simple else len(list(cat_list.keys()))

cfg.MODEL.ROI_HEADS.NUM_CLASSES = (
    2 if simple else num_classes  # former 5
)  # we have the malign and benign class / all 5 classes


# Color and Class definitions

mal_col = (0, 69, 255)  # red
ben_col = (50, 205, 50)  # green

if simple:
    MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_colors = [
        ben_col,
        mal_col,
    ]  # Benign(green), Malignant(red)
elif advanced:
    MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_colors = [
        # maligne
        mal_col,
        mal_col,
        mal_col,
        mal_col,
        mal_col,
        # benigne
        ben_col,
        ben_col,
        ben_col,
        ben_col,
        ben_col,
        ben_col,
        ben_col,
        ben_col,
        ben_col,
        ben_col,
        ben_col,
    ]
else:
    MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_colors = [
        # maligne
        mal_col,
        mal_col,
        mal_col,
        # benigne
        ben_col,
        ben_col,
    ]


# Select Trainer
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

if train:
    trainer = ud.CocoTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


print(model_str)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_str)
# set the testing threshold for this model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
cfg.DATASETS.TEST = ("my_dataset_test",)
predictor = DefaultPredictor(cfg)

# get the shuffled indexes
dis = get_advanced_dis_df(df)
dis_ex = get_advanced_dis_df(df_ex, mode=True)
d = [os.path.join("./PNG2", f"{f}.png") for f in df[F_KEY]]
d_ex = [os.path.join("./PNG_external", f"{f}.png")
        for f in df_ex['id']]

# get the active indexes for each dataset
train_idx = dis["train"]["idx"]
valid_idx = dis["valid"]["idx"]
test_idx = dis["test"]["idx"]
text_ex_idx = dis_ex["test_external"]



# %% Evaluate basic results:

res = ud.personal_score(predictor, df, simple=simple, mode="test")

print(f'Entities: {res["acc"]}, Benign/Malign:{res["acc2"]}')
norm = False
label = (
    ["Benign", "Malignant"]
)
ud.plot_confusion_matrix(
    res["conf2"], ["Benign", "Malignant"], normalize=norm
) if not simple else None
ud.plot_confusion_matrix(
    res["conf"], label, normalize=norm,
)
ud.plot_roc_curve(res['rocauc'][0], res['rocauc'][1], res['rocauc'][2])


res = ud.personal_score(predictor, df_ex, simple=simple,
                        mode="test_external", external=True, imgpath='./PNG_external')
print(f'Entities: {res["acc"]}, Benign/Malign:{res["acc2"]}')
norm = False
label = (
    ["Benign", "Malignant"]
)
ud.plot_confusion_matrix(
    res["conf2"], ["Benign", "Malignant"], normalize=norm
) if not simple else None
ud.plot_confusion_matrix(
    res["conf"], label, normalize=norm,
)


# %% View all results as interactive widget

idx = widgets.IntSlider(min=0, max=len(test_idx) - 1,
                        value=0, description="Index:")
proposed = widgets.IntSlider(
    min=0, max=10, value=1, description="Number of Boxes:")
widgets.interactive(update, idx=idx, proposes=proposed)


# %% Perform scoring

res = ud.personal_score(predictor, df, simple=simple, mode="test")
print(f'Entities: {res["acc"]}, Benign/Malign:{res["acc2"]}')
dh.plot_roc_curve(res['rocauc'][0], res['rocauc'][1], res['rocauc'][2])

_, cat_mapping = make_categories_advanced(
    simple) if advanced else make_categories(simple)


norm = False
label = (
    ["Benign", "Malignant"]
    if simple
    else cat_mapping.keys()
)
ud.plot_confusion_matrix(
    res["conf2"], ["Benign", "Malignant"], normalize=norm
) if not simple else None
ud.plot_confusion_matrix(
    res["conf"], label, normalize=norm,
)


# %% Print the IoU and dice-scores

    
dh.print_iou_dice_scores(predictor, df)


# %% Generate all images with the trained model

dh.generate_all_images(external=True)

# %%

res = dh.personal_advanced_score(
    predictor, df, mode="test", simple=False, imgpath="./PNG2", advanced=True)

for i, cat in enumerate(res.keys()):
    fig = ud.plot_confusion_matrix(
        res[cat]["conf"], cat_naming_new[i]['catnames'], normalize=False)

# %%

iou_mask, dice_mask, iou_bb, dice_bb = dh.get_iou_masks(external=False)
iou_mask, dice_mask, iou_bb, dice_bb = dh.get_iou_masks(external=True)

# %% Perform Evaluation on internal and external dataset

df_ex, paths_ex = get_df_paths(mode=True)

res = ud.personal_score(predictor, df, simple=simple, mode='test', external=False)
res_ex = ud.personal_score(predictor, df_ex, simple=simple,
                        mode="test_external", external=True, imgpath='./PNG_external')

# %% Inernal Conf Matrix
ud.plot_confusion_matrix(res['conf'], ['Benign', 'Malignant'] if simple else reverse_cat_list, ft=24)

# %% External Conf MAtrix
ud.plot_confusion_matrix(res_ex['conf'], ['Benign', 'Malignant'] if simple else reverse_cat_list, ft=24)
# %% Internal Conf Info

dh.print_confinfo(res['conf'])

# %% External Conf Info
dh.print_confinfo(res_ex['conf'])


# %% Internal
preds = [loc.numpy() for loc in res['preds']]
targets = res['targets']
kappa = cohen_kappa_score(preds, targets)
dh.get_ci(kappa, len(preds))

sumit = 0
for pred, target in zip(preds, targets):
    if pred == target:
        sumit += 1
print(f'{sumit} of {len(preds)}')
dh.get_ci(res['acc'], len(preds))


# %% External
preds = [loc.numpy() for loc in res_ex['preds']]
targets = res_ex['targets']
kappa = cohen_kappa_score(preds, targets)
dh.get_ci(kappa, len(preds))

sumit = 0
for pred, target in zip(preds, targets):
    if pred == target:
        sumit += 1
print(f'{sumit} of {len(preds)}')
dh.get_ci(res_ex['acc'], len(preds))
# %%
