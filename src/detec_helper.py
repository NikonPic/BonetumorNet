# %%
from PIL import Image, ImageDraw, ImageFont, ImageOps
import sys
import nrrd
import cv2
from sklearn.metrics import confusion_matrix
from tqdm.notebook import tqdm
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import ColorMode

from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import torch
import torchvision
from src.utils_tumor import dis_df, get_df_paths, get_cocos_from_df, get_advanced_dis_df, format_seg_names
from src.utils_detectron import F_KEY, CLASS_KEY, ENTITY_KEY
from src.categories import make_categories_advanced, make_categories, cat_mapping_new, cat_naming_new, reverse_cat_list
#from src.utils_detectron import get_dataloader, personal_score, plot_confusion_matrix, plot_roc_curve, eval_iou_dice, personal_score_simple
import src.utils_detectron as ud


# import some common libraries
import numpy as np
import random
import os
import detectron2
from detectron2.utils.logger import setup_logger
from sklearn.metrics import cohen_kappa_score

setup_logger()
cfg = get_cfg()
print(detectron2.__version__)

#  function definitions for training and evaluation

def get_mask_img(df, idx, scale=1, truelab='blue', external=False):
    """extract the segmented image and put it on the image"""
    df_loc = df
    segpath_loc = './SEG'

    if external:
        df_loc = df_ex
        segpath_loc = './SEG_external'

    filename = df_loc[F_KEY][idx]
    filename_seg = format_seg_names(filename)
    nrrd_file = nrrd.read(f'{segpath_loc}/{filename_seg}.seg.nrrd')
    nrrd_arr = np.transpose(np.array(nrrd_file[0])[:, :, 0] * 90)

    offset_strings = nrrd_file[1]['Segmentation_ReferenceImageExtentOffset'].split(' ')[
        :2]
    offset = [int(off) for off in offset_strings]

    im = Image.fromarray(nrrd_arr, mode='L').convert('L')
    im_mask = im.copy()
    im = ImageOps.colorize(im, black='black', white=truelab)

    # now rescale:
    return im, offset, im_mask


def update(
    idx=1, bbox=True, mask=True, score=True, scale=1, true_label=True
):
    """Display the activations"""
    proposed = 1
    mode = 'test'

    if mode == "test":
        active_idx = test_idx
    elif mode == "valid":
        active_idx = valid_idx
    else:
        active_idx = train_idx

    im = cv2.imread(d[active_idx[idx]])

    with torch.no_grad():
        outputs = predictor(im)

    v = Visualizer(
        im[:, :, ::-1],
        metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
        scale=scale,
        instance_mode=ColorMode.SEGMENTATION,
    )
    instances = outputs["instances"].to("cpu")[:proposed]
    instances.remove("pred_masks") if not mask else None
    instances.pred_boxes = [[0, 0, 0, 0]] if not bbox else instances.pred_boxes
    instances.remove("scores") if not score else None

    v = v.draw_instance_predictions(instances)
    plt.figure(figsize=(8, 8))

    if df[CLASS_KEY][active_idx[idx]] == 1:
        labelcol = 'blue'
    else:
        labelcol = 'red'

    im, offset, im_mask = get_mask_img(
        df, idx=active_idx[idx], truelab=labelcol)

    back_img = Image.fromarray(v.get_image()[:, :, ::-1])
    back_img.paste(im, offset, im_mask) if true_label else None
    back_img = np.array(back_img)
    Image.fromarray(back_img).show()

    print(
        "Malign!"
        if df[CLASS_KEY][active_idx[idx]] == 1
        else "Benign!"
    )
    print(df["Tumor.Entitaet"][active_idx[idx]])

    return back_img


def plot_thresh_iou(ious):
    thresh = np.linspace(start=0, stop=1, num=100)

    res = []
    for th in thresh:
        res.append(
            sum([True if iou > th else False for iou in ioUs_box]) / len(ioUs_box))

    plt.figure(figsize=(12, 12))
    plt.grid(0.25)
    plt.plot(thresh, res)
    plt.xlabel("IoU")
    plt.ylabel("Accuracy")


def generate_all_images(external=False):
    """Display the activations"""
    proposed = 1
    mode = 'test'
    scale = 1
    mask = True
    bbox = True
    score = True

    if mode == "test":
        active_idx = test_idx
    elif mode == "valid":
        active_idx = valid_idx
    else:
        active_idx = train_idx

    d_loc = d
    add_str = 'normal'
    df_loc = df

    if external:
        active_idx = text_ex_idx['idx']
        d_loc = d_ex
        df_loc = df_ex

        add_str = 'external'

    for idx, _ in tqdm(enumerate(active_idx)):
        im = cv2.imread(d_loc[active_idx[idx]])
        im_org = Image.fromarray(im)
        pngname = df_loc[F_KEY][active_idx[idx]]
        im_org.save(f'./res/{add_str}/{pngname}.png')

        with torch.no_grad():
            outputs = predictor(im)

        v = Visualizer(
            im[:, :, ::-1],
            metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            scale=scale,
            instance_mode=ColorMode.SEGMENTATION,
        )
        instances = outputs["instances"].to("cpu")[:proposed]
        instances.remove("pred_masks") if not mask else None
        instances.pred_boxes = [[0, 0, 0, 0]
                                ] if not bbox else instances.pred_boxes
        instances.remove("scores") if not score else None

        v = v.draw_instance_predictions(instances)
        plt.figure(figsize=(8, 8))

        if df_loc[CLASS_KEY][active_idx[idx]] == 1:
            labelcol = 'red'
        else:
            labelcol = 'blue'

        im, offset, im_mask = get_mask_img(
            df, idx=active_idx[idx], truelab=labelcol, external=external)

        back_img = Image.fromarray(v.get_image()[:, :, ::-1])
        back_img.paste(im, offset, im_mask)
        back_img = np.array(back_img)
        img = Image.fromarray(back_img)

        img.save(f'./res/{add_str}/{pngname}_annotated.png')


def personal_advanced_score(predictor, df, mode="test", simple=False, imgpath="./PNG2", advanced=True):
    """define the accuracy"""
    # get the dataset distribution
    active_idx = test_idx

    # get the actibe files
    files = [os.path.join(imgpath, f"{f}.png") for f in df[F_KEY]]

    # apply the category mapping dep. on simple-mode
    _, cat_mapping = make_categories_advanced(simple)

    res = {}

    for loc_cat in cat_naming_new:

        cat_index = loc_cat['index']
        cat_name = loc_cat['name']
        res[cat_name] = {}

        # counters during evaluation
        count = 0

        # to be filled arrays
        preds, targets = [], []

        # Go over the whole dataset
        for idx in tqdm(active_idx):

            # load image
            im = cv2.imread(files[idx])

            # get predicitions
            with torch.no_grad():
                outputs = predictor(im)
                out = outputs["instances"].to("cpu")
            pred_entity_int = out[:1].pred_classes[0]
            pred_entity_str = reverse_cat_list[pred_entity_int]

            # now get the mapped local prediciton
            loc_pred = cat_mapping_new[pred_entity_str][cat_index]
            preds.append(loc_pred)

            # get the true output
            true_entity_str = df[ENTITY_KEY][idx]
            loc_target = cat_mapping_new[true_entity_str][cat_index]
            targets.append(loc_target)

            # increase count
            count += 1 if loc_target == loc_pred else 0

        conf = confusion_matrix(targets, preds)
        acc = count / len(active_idx)
        print(f'ACC {cat_name}: {round(acc, 3)}')

        res[cat_name] = {
            'conf': conf,
            'acc': acc
        }

    return res


def make_bool(x):
    """turn an array of ints into array of bool"""
    if x > 0:
        return True
    return False

vfunc = np.vectorize(make_bool)

def make_1(x):
    if x:
        return 1
    return 0

vfunc1 = np.vectorize(make_1)  

def compare_masks(mask1, mask2):
    """calculate iou and dice score with mask1 and mask2"""
    overlap = mask1*mask2 # Logical AND
    intersection = np.sum(overlap)
    union = mask1 + mask2 # Logical OR

    iou = overlap.sum()/float(union.sum() - overlap.sum()) # Treats "True" as 1,
                                       # sums number of Trues
                                       # in overlap and union
                                       # and divides
    dice = np.mean((2. * intersection)/float(union.sum()))
    return iou, dice

def get_bb_from_mask(mask):
    """take max and min from mask in x and y direction"""
    mask_bb = mask.copy()
    sh = mask.shape
    min_y = sh[0]
    max_y = 0

    min_x = sh[1]
    max_x = 0

    for row in range(sh[0]):
        if True in mask[row, :]:
            max_y = row
            if row < min_y:
                min_y = row

    for column in range(sh[1]):
        if True in mask[:, column]:
            max_x = column
            if column < min_x:
                min_x = column
    
    mask_bb[min_y:max_y+1, min_x:max_x+1] = True
    
    return mask_bb

def get_iou_masks(external=False):
    """Display the activations"""
    mode = 'test'
    scale = 1
    mask = True
    bbox = True

    if mode == "test":
        active_idx = test_idx
    elif mode == "valid":
        active_idx = valid_idx
    else:
        active_idx = train_idx

    d_loc = d
    add_str = 'normal'
    df_loc = df

    iou_all_mask = []
    dice_all_mask = []

    iou_all_bb = []
    dice_all_bb = [] 

    if external:
        active_idx = text_ex_idx['idx']
        d_loc = d_ex
        df_loc = df_ex

        add_str = 'external_1'

    for idx, i in tqdm(enumerate(active_idx)):
        im = cv2.imread(d_loc[active_idx[idx]])
        im_org = Image.fromarray(im)
        pngname = df_loc[F_KEY][active_idx[idx]]
        im_org.save(f'./res/{add_str}/{pngname}.png')

        with torch.no_grad():
            outputs = predictor(im)

            v = Visualizer(
                im[:, :, ::-1],
                metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                scale=scale,
                instance_mode=ColorMode.SEGMENTATION,
            )

            instances = outputs["instances"].to("cpu")[:1]
            instances.remove("pred_masks") if not mask else None
            instances.pred_boxes = [[0, 0, 0, 0]] if not bbox else instances.pred_boxes

        v = v.draw_instance_predictions(instances)

        mask_pred = instances.pred_masks[0].numpy()
        mask_pred_bb = get_bb_from_mask(mask_pred)

        mask_pred = vfunc1(mask_pred)
        mask_pred_bb = vfunc1(mask_pred_bb)

        im, offset, im_mask = get_mask_img(
                    df, idx=active_idx[idx], truelab='red', external=external)

        back_img = Image.fromarray(v.get_image()[:, :, ::-1] * 0)
        back_img.paste(im, offset, im_mask)
        back_img = np.array(back_img)

        mask_true = vfunc(back_img[:, :, 0])
        mask_true_bb = get_bb_from_mask(mask_true)

        mask_true = vfunc1(mask_true)
        mask_true_bb = vfunc1(mask_true_bb)

        iou_loc, dice_loc = compare_masks(mask_pred, mask_true)
        iou_loc_bb, dice_loc_bb = compare_masks(mask_pred_bb, mask_true_bb)

        iou_all_mask.append(iou_loc)
        dice_all_mask.append(dice_loc)

        iou_all_bb.append(iou_loc_bb)
        dice_all_bb.append(dice_loc_bb)

    corr_mask = sum([1 if iou > 0.5 else 0 for iou in iou_all_mask])
    corr_bb = sum([1 if iou > 0.5 else 0 for iou in iou_all_bb])

    p_mode = 'External' if external else 'Internal'
    print(f'Mode: {p_mode}')
    print(f'Corr Masks: {corr_mask}, {round(corr_mask/len(iou_all_mask), 2)}')
    print(f'Corr BB: {corr_bb}, {round(corr_bb/len(iou_all_bb), 2)}')

    print(f'Iou  Mask: {round(np.mean(iou_all_mask), 2)} +/- {round(np.std(iou_all_mask), 2)}')
    print(f'Dice Mask: {round(np.mean(dice_all_mask), 2)} +/- {round(np.std(dice_all_mask), 2)}')

    print(f'Iou  BB: {round(np.mean(iou_all_bb), 2)} +/- {round(np.std(iou_all_bb), 2)}')
    print(f'Dice BB: {round(np.mean(dice_all_bb), 2)} +/- {round(np.std(dice_all_bb), 2)}')

    return iou_all_mask, dice_all_mask, iou_all_bb, dice_all_bb


def get_ci(acc, n=140, const=1.96, digits=3, printit=True):
    """calculate confidence intervall"""
    acc = round(acc, digits)
    error = 1 - acc
    ci_low  = round((1 - (error - const * np.sqrt((error * (1 - error)) / n))), digits) * 100
    ci_high = round((1 - (error + const * np.sqrt((error * (1 - error)) / n))), digits) * 100
    if printit:
        print(f'Acc: {acc*100}%, 95% CI: {ci_high}%, {ci_low}%)')
    return ci_high, ci_low

def print_confinfo(conf):
    """print all relevant infos for confidence intervalls"""
    tp = conf[1, 1]
    tn = conf[0, 0]
    fp = conf[0, 1]
    fn = conf[1, 0]

    sens = round(tp / (tp + fn), 3)
    sens_high, sens_low = get_ci(sens, n=tp+fn, printit=False)
    spec = round(tn / (tn + fp), 3)
    spec_high, spec_low = get_ci(spec, n=tn+fp, printit=False)
    acc  = round((tn + tp) / (tn + fp + tp + fn), 3)
    acc_high, acc_low = get_ci(acc, n=tn + fp + tp + fn, printit=False)


    print(f'sensitivity : {sens} ({tp} of { (tp + fn)}), 95% CI: {sens_high}% {sens_low}%')
    print(f'specificity : {spec} ({tn} of { (tn + fp)}), 95% CI: {spec_high}% {spec_low}%')
    print(f'accuracy : {acc} ({tn + tp} of { (tn + fp + tp + fn)}), 95% CI: {acc_high}% {acc_low}%')

def evaluate(dset):
    """Use the detectron coco evaluator"""

    evaluator = ud.COCOEvaluator(dset, cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, dset)
    res = ud.inference_on_dataset(predictor.model, val_loader, evaluator)

    return res

def print_iou_dice_scores(predictor, df):
    """print the dice scores and iou results"""
    ioUs_box, dices_box, ioUs_mask, dices_mask = ud.eval_iou_dice(
        predictor, df, proposed=1, mode="test")
    print('BBOX:')
    print(f'IoU: {np.mean(ioUs_box)} +/- {np.std(ioUs_box)}')
    print(f'Dice: {np.mean(dices_box)} +/- {np.std(dices_box)}\n')

    print('SEG:')
    print(f'IoU: {np.mean(ioUs_mask)} +/- {np.std(ioUs_mask)}')
    print(f'Dice: {np.mean(dices_mask)} +/- {np.std(dices_mask)}')