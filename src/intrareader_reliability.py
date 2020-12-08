# %% Intrareader reliability
import os
import nrrd
import numpy as np
from PIL import Image
from tqdm import tqdm


p1 = 'SEG'
p2 = 'Intrareader_seg'
im_p = 'PNG2'

all_segs = os.listdir(p1)
comp_segs = os.listdir(p2)


def get_nrrd_mask(filename, img_path, nrrd_path, fac=15, nrrd_key='Segmentation_ReferenceImageExtentOffset'):
    # load nrrd image
    readdata, header = nrrd.read(f'{nrrd_path}/{filename}')
    nrrd_img = np.transpose(readdata[:, :, 0] * fac)

    # get the offsets
    offset = header[nrrd_key].split()
    offset = [int(off) for off in offset]
    offset = offset[0:2]

    # load true image
    background = Image.open(f'{img_path}/{filename[:-9]}.png')
    foreground = Image.fromarray(nrrd_img)

    # generate masked image
    mask = Image.fromarray(np.array(background) * 0)
    mask.paste(foreground, offset, foreground)

    return np.array(mask)[:, :, 0]

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

def make_bool(x):
    if x > 0:
        return True
    return False

def make_1(x):
    if x:
        return 1
    return 0


def get_bb_from_mask(mask):
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
    
    return vfunc1(mask_bb)

vfunc = np.vectorize(make_bool)
vfunc1 = np.vectorize(make_1)



comp2_seg = []

# ensure all files exist
for seg in all_segs:
    if seg in comp_segs:
        comp2_seg.append(seg)


ious_bb = []
dice_scores_bb = []

ious_mask = []
dice_scores_mask = []

# go trough all segmentations
for seg in tqdm(comp2_seg):
    mask1 = get_nrrd_mask(seg, im_p, p1)
    mask2 = get_nrrd_mask(seg, im_p, p2)

    mask1_bb = get_bb_from_mask(mask1)
    mask2_bb = get_bb_from_mask(mask2)

    mask1 = vfunc1(mask1)
    mask2 = vfunc1(mask2)

    iou_loc, dice_loc = compare_masks(mask1, mask2)
    iou_loc_bb, dice_loc_bb = compare_masks(mask1_bb, mask2_bb)
    
    ious_bb.append(iou_loc_bb)
    dice_scores_bb.append(dice_loc_bb)

    ious_mask.append(iou_loc)
    dice_scores_mask.append(dice_loc)


# %%
iou_mean, iou_std = np.array(ious_mask).mean().round(decimals=2), np.array(ious_mask).std().round(decimals=3)
dice_mean, dice_std = np.array(dice_scores_mask).mean().round(decimals=2), np.array(dice_scores_mask).std().round(decimals=2)
print(f'IOU MASK: {iou_mean} +/- {iou_std}')
print(f'DICE MASK: {dice_mean} +/- {dice_std}')


iou_mean, iou_std = np.array(ious_mask).mean().round(decimals=2), np.array(ious_mask).std().round(decimals=2)
dice_mean, dice_std = np.array(dice_scores_mask).mean().round(decimals=2), np.array(dice_scores_mask).std().round(decimals=2)
print(f'IOU BB: {iou_mean} +/- {iou_std}')
print(f'DICE BB: {dice_mean} +/- {dice_std}')
# %%

seg = comp2_seg[0]
mask1 = get_nrrd_mask(seg, im_p, p1)
mask2 = get_nrrd_mask(seg, im_p, p2)