
# %%
#
#  utils_tumor.py
#  BonetumorNet
#
#  Created by Nikolas Wilhelm on 2020-08-01.
#  Copyright © 2020 Nikolas Wilhelm. All rights reserved.
#

# general
import os
import json
from datetime import datetime, date
from itertools import groupby
import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
import nrrd
# open images
from PIL import Image, ImageFile

# generate polygons:
from imantics import Mask

# import personal functions
if __name__ == '__main__':++
    from categories import make_cat_advanced, cat_mapping_new, reverse_cat_list, malign_int
else:
    from src.categories import make_cat_advanced, cat_mapping_new, reverse_cat_list, malign_int


ImageFile.LOAD_TRUNCATED_IMAGES = True

# %% Create Databunch functionality

# Constant definitions to avoid repetitions :
VALID_PART = 0.15
TEST_PART = 0.15
SEED = 53
np.random.seed(SEED)

F_KEY = 'FileName (png)'
CLASS_KEY = 'Aggressiv/Nicht-aggressiv'
ENTITY_KEY = 'Tumor.Entitaet'


def get_advanced_dis_data_fr(data_fr_loc, mode=False):
    """
    redefine the dataframe distribution for advanced training -> separate by entities!
    """
    # 1. get number of entities in overall_data_fr_loc
    # 2. split entities according to train, val / test-split

    # init the empyt idx lists
    train_idx = []
    valid_idx = []
    test_idx = []

    # get the categories by which to split
    cats = reverse_cat_list

    for cat in cats:
        # get all matching data_fr_loc entries
        data_fr_loc_loc = data_fr_loc.loc[data_fr_loc[ENTITY_KEY] == cat]
        loclen = len(data_fr_loc_loc)

        # now split acc to the indices
        validlen = round(loclen * VALID_PART)
        testlen = round(loclen * TEST_PART)
        trainlen = loclen - validlen - testlen

        # get the matching indices and extend the idx list
        data_fr_loc_loc_train = data_fr_loc_loc.iloc[:trainlen]
        train_idx.extend(list(data_fr_loc_loc_train.index))

        data_fr_loc_loc_valid = data_fr_loc_loc.iloc[trainlen:trainlen+validlen]
        valid_idx.extend(list(data_fr_loc_loc_valid.index))

        data_fr_loc_loc_test = data_fr_loc_loc.iloc[trainlen+validlen::]
        test_idx.extend(list(data_fr_loc_loc_test.index))

    # summarize in dictionary
    dis = {
        'train': {
            'len': len(train_idx),
            'idx': train_idx,
        },
        'valid': {
            'len': len(valid_idx),
            'idx': valid_idx,
        },
        'test': {
            'len': len(test_idx),
            'idx': test_idx,
        }
    }

    if mode:
        dis = {
            'test_external': {
                'len': len(data_fr_loc),
                'idx': list(range(len(data_fr_loc))),
            }
        }

    return dis


def calculate_age(born, diag):
    """get the age from the calendar dates"""
    born = datetime.strptime(born, "%d.%m.%Y").date()
    diag = datetime.strptime(diag, "%d.%m.%Y").date()
    return diag.year - born.year


def apply_cat(train, valid, test, dis, new_name, new_cat):
    """add a new category to the dataframe"""
    train_idx = dis['train']['idx']
    valid_idx = dis['valid']['idx']
    test_idx = dis['test']['idx']

    train[new_name] = [new_cat[idx] for idx in train_idx]
    valid[new_name] = [new_cat[idx] for idx in valid_idx]
    test[new_name] = [new_cat[idx] for idx in test_idx]
    return train, valid, test


def get_data_fr_dis(data_fr_loc, born_key='OrTBoard_Patient.GBDAT', diag_key='Erstdiagnosedatum',
                    t_key='Tumor.Entitaet', pos_key='Befundlokalisation', out=True,
                    mode=False):
    """
    extract ages and other information from data_fr_loc
    """

    # get ages
    if mode:
        ages_loc = data_fr_loc['Alter bei Erstdiagnose']
    else:
        ages_loc = [calculate_age(born, diag) for (born, diag) in zip(
            data_fr_loc[born_key], data_fr_loc[diag_key])]

    # get labels
    labels = [float(lab) for lab in data_fr_loc[CLASS_KEY]]

    # get male(0) / female(1)
    if mode:
        female_male = [1 if d_loc ==
                       'f' else 0 for d_loc in data_fr_loc['Geschlecht']]
    else:
        female_male = [int(name[0] == 'F') for name in data_fr_loc[F_KEY]]

    # tumor_kind
    tumor_kind = data_fr_loc[t_key]

    # position
    position = data_fr_loc[pos_key]

    # get the shuffled indexes
    dis = get_advanced_dis_data_fr(data_fr, mode=mode)

    if out:
        for key in dis.keys():
            print(f"{key}:")
            print_info(ages_loc, labels, female_male,
                       dis[key]['idx'], tumor_kind, position)

        print("All:")
        print_info(ages_loc, labels, female_male, list(
            range(len(ages_loc))), tumor_kind, position)

    return ages_loc


def print_info(loc_ages, labels, female_male, active_idx, tumor_kind, position, nums=1):
    """
    summarize all informations as a print message
    """

    age = np.array([loc_ages[i] for i in active_idx]).mean().round(nums)
    age_std = np.array([loc_ages[i]
                        for i in active_idx]).std().round(nums)
    print(f'Age: {age} ± {age_std}')

    females = np.array([female_male[i] for i in active_idx]).sum()
    femals_p = round((100*females) / len(active_idx), nums)
    print(f'Female: {females} ({femals_p}%)')

    malign = int(np.array([labels[i] for i in active_idx]).sum())
    malign_p = round((100 * malign) / len(active_idx), nums)
    print(f'Malignancy: {malign} ({malign_p}%)')
    print(f'Benign: {len(active_idx)-malign} ({100-malign_p}%)')

    _, cat_mapping = make_cat_advanced(simple=False)

    tumor_list = list(cat_mapping.keys())

    for tumor in tumor_list:
        tums = [int(tumor == name)
                for name in tumor_kind[active_idx]]
        num_tums = np.array(tums).sum()
        per_tum = round(100 * num_tums / len(active_idx), nums)
        print(f'{tumor}: {num_tums} ({per_tum}%)')

    position_dict = {}
    position_dict['Torso/head'] = ['Becken',
                                   'Thoraxwand', 'Huefte', 'LWS', 'os sacrum']
    position_dict['Upper Extremity'] = [
        'Oberarm', 'Hand', 'Schulter', 'Unterarm']
    position_dict['Lower Extremity'] = [
        'Unterschenkel', 'Fuß', 'Knie', 'Oberschenkel']

    for pos_k in position_dict.keys():
        cur_pos = [int(p in position_dict[pos_k])
                   for p in position[active_idx]]
        num_pos = np.array(cur_pos).sum()
        per_pos = round(100 * num_pos / len(active_idx), nums)
        print(f'{pos_k}: {num_pos} ({per_pos}%)')

    dset_part = round(100 * len(active_idx) / len(loc_ages), nums)
    print(f'Dataset Nums: {len(active_idx)} ({dset_part}%)\n\n')

# %% Preparation: Create the coco format


def add_classes_to_csv(csv_path, mode=False):
    """
    construct the bounding boxes and add them to the csv file
    """
    # open csv
    if mode:
        data_fr_loc = pd.read_excel(csv_path)
    else:
        data_fr_loc = pd.read_csv(csv_path, header='infer', delimiter=';')

    len_data_fr = len(data_fr_loc)

    # predefine arrays
    agg_non_agg, ben_loc_mal, clinicla_flow, clinicla_flow_red, super_ent = np.empty(
        [len_data_fr]), np.empty([len_data_fr]), np.empty([len_data_fr]), np.empty(
        [len_data_fr]), np.empty([len_data_fr])

    # iterate trough the files:
    for i, label in tqdm(enumerate(data_fr_loc[ENTITY_KEY])):
        agg_non_agg[i] = cat_mapping_new[label][1]
        ben_loc_mal[i] = cat_mapping_new[label][2]
        clinicla_flow[i] = cat_mapping_new[label][3]
        clinicla_flow_red[i] = cat_mapping_new[label][4]
        super_ent[i] = cat_mapping_new[label][6]

    benmal_info = []
    for loc_ent in data_fr_loc[ENTITY_KEY]:
        ent_int = cat_mapping_new[loc_ent][0]
        benmal = 1 if ent_int in malign_int else 0
        benmal_info.append(benmal)

    long_txt = 'Grade for clinical workflow (2 + 3 = 2 > assessment in MSK center needed)'

    # add the bounding boxes to the dataframe
    data_fr_loc[CLASS_KEY] = benmal_info
    data_fr_loc[long_txt] = clinicla_flow_red

    data_fr_loc['Aggressive - Non Aggressive'] = agg_non_agg
    data_fr_loc['Benigne - Local Aggressive - Maligne'] = ben_loc_mal
    data_fr_loc['Grade of clinical workflow'] = clinicla_flow
    data_fr_loc['Super Entity (chon: 0, osteo:1, meta:2, other:3)'] = super_ent

    # save to csv!
    if mode:
        data_fr_loc.to_excel(csv_path)
    else:
        data_fr_loc.to_csv(csv_path, sep=';', index=False)

    return data_fr_loc


def nrrd_2_mask(nrrd_path, im_path, nrrd_key='Segmentation_ReferenceImageExtentOffset',
                fac=20, as_array=False):
    """
    generate mask from the nrrd file
    """
    # load nrrd image
    readdata, header = nrrd.read(nrrd_path)
    nrrd_img = np.transpose(readdata[:, :, 0] * fac)

    # get the offsets
    offset = header[nrrd_key].split()
    offset = [int(off) for off in offset]
    offset = offset[0:2]

    # load true image
    background = Image.open(im_path)
    foreground = Image.fromarray(nrrd_img)

    # generate masked image
    mask = Image.fromarray(np.array(background) * 0)
    mask.paste(foreground, offset, foreground)

    return np.array(mask)[:, :, 0] if as_array else mask


def format_seg_names(name):
    """replace 'ö','ä','ü',',',' '  """
    name = name if name[-1] != ' ' else name[:-1]
    name = name.replace('ö', 'oe').replace('Ö', 'OE')
    name = name.replace('ä', 'ae').replace('Ä', 'AE')
    name = name.replace('ü', 'ue').replace('Ü', 'UE')
    name = name.replace(',', '')
    return name


def generate_masks(data_frame, nrrd_path, pic_path, mask_path, gen_rad=True):
    """
    save all pictures in a masked version in the mask_folder
    """

    for file in tqdm(data_frame[F_KEY]):
        # get names
        pic_name = os.path.join(pic_path, f'{file}.png')
        file = format_seg_names(file)
        nrrd_name = os.path.join(nrrd_path, f'{file}.seg.nrrd')

        # mask the picture
        try:
            mask = nrrd_2_mask(nrrd_name, pic_name, fac=255)

            # save masked picture:
            mask_name = os.path.join(mask_path, f'{file}.png')
            mask.save(mask_name)

            if gen_rad:
                mask = np.array(mask)
                shape = mask.shape
                mask = mask.reshape((shape[2], shape[1], shape[0]))
                nrrd.write(f'../radiomics/label/{file}.nrrd', mask)

        except FileNotFoundError:
            print(nrrd_name)

# %% coco-formatting


def make_empty_coco(mode='train', simple=True):
    des = f'{mode}-BoneTumor detection in coco-format'
    today = date.today()
    today_str = str(today.year) + str(today.month) + str(today.day)
    cat_list, cat_mapping = make_cat_advanced(simple)

    coco = {
        "infos": {
            "description": des,
            "version": "0.01",
            "year": today.year,
            "contributor": "Nikolas Wilhelm",
            "date_created": today_str
        },
        "licences": [
            {
                "id": 1,
                "name": "todo"
            },
        ],
        "categories": cat_list,
        "images": [],
        "annotations": [],
    }
    return coco, cat_mapping


def get_cocos_from_data_fr(data_fr_loc, paths_loc, save=True, simple=True, newmode=0, ex_mode=False):
    """
    build the coco dictionaries from the dataframe
    """
    # get the shuffled indexes
    dis = get_advanced_dis_data_fr(data_fr_loc, mode=ex_mode)

    # the list of coco dictionaries
    cocos_loc = []

    for i, mode in enumerate(dis.keys()):
        # get the active indices
        indices = dis[mode]['idx']

        # make empty coco_dict
        cocos_loc.append(make_coco(data_fr_loc, mode, indices, newmode=newmode,
                                   simple=simple, path=paths_loc["pic"], path_nrd=paths_loc["seg"]))

        if save:
            local_path = os.getcwd()
            add = "../" if local_path[-3:] == "src" else ""

            save_file = f'{add}{mode}.json'
            print(f'Saving to: {save_file}')
            with open(save_file, 'w') as file_p:
                json.dump(cocos_loc[i], file_p, indent=2)

    return cocos_loc


def bbox_from_segm(segm):
    """create bounding box"""
    seg = segm[0]
    x_arr = seg[0::2]
    y_arr = seg[1::2]

    top = min(y_arr)
    left = min(x_arr)
    bottom = max(y_arr)
    right = max(x_arr)
    return tlbr2bbox(top, left, bottom, right)


def tlbr2bbox(top, left, bottom, right, oper=int):
    """
    tlbr = [top, left, bottom, right]
    to ->
    bbox = [x(left), y(top), width, height]
    """
    x_pos = oper(left)
    y_pos = oper(top)
    width = oper(right - left)
    height = oper(bottom - top)

    return [x_pos, y_pos, width, height]


def check_seg(segl):
    """check the segmentation format -> take the largest segmentation"""
    checked = segl.copy()

    # take the longest if we have multiple polygons ..?
    if len(segl) > 1:
        maxlen = 0
        for loc_seg in segl:
            if len(loc_seg) > maxlen:
                maxlen = len(loc_seg)
                checked = [loc_seg]

    return checked


def make_coco(data_frame, mode, idxs, path='../PNG2', path_nrd='../SEG', simple=True, newmode=0):
    """fill the coco with the annotations"""

    # create the empty coco format
    coco, cat_mapping = make_empty_coco(mode, simple=simple)

    # go trough all indexes and append the img-names and annotations
    for idx in idxs:

        # get the current object
        obj = data_frame.iloc[idx]

        # get the filename
        file = obj[F_KEY]
        filename = file + '.png'

        # get height and width by loading the picture
        filepath = os.path.join(path, filename)
        img_s = np.array(Image.open(filepath)).shape
        height, width = img_s[0], img_s[1]

        # get the image id -> idx should be unique
        id_tumor = int(idx)

        # get the class:
        name = obj[CLASS_KEY] if simple else obj[ENTITY_KEY]

        cat = cat_mapping[name]

        if newmode > 0:
            cat = cat_mapping_new[name][newmode]

        # build the image dictionary
        img_dict = {
            "id": id_tumor,
            "file_name": filename,
            "height": height,
            "width": width,
        }

        # build the annotation dictionary
        ann_dict = {
            "id": id_tumor,
            "image_id": id_tumor,
            "category_id": cat,
            "iscrowd": 0,
            "area": int(height * width),
        }

        # get the segmentation
        segname = format_seg_names(file)

        # get the rle - mask
        nrrdpath = os.path.join(path_nrd, segname + '.seg.nrrd')
        mask = nrrd_2_mask(nrrdpath, filepath, as_array=True)
        polygons = Mask(mask).polygons()

        ann_dict["area"] = int(np.sum(mask > 0))
        segm = check_seg(polygons.segmentation)
        ann_dict["segmentation"] = segm
        ann_dict['bbox'] = bbox_from_segm(segm)

        # append the dictionaries to the coco bunch
        coco['images'].append(img_dict)
        coco['annotations'].append(ann_dict)

    return coco


def binary_mask_to_rle(binary_mask):
    """
    https://stackoverflow.com/questions/49494337/encode-numpy-array-using-uncompressed-rle-for-coco-dataset
    """
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


def get_data_fr_paths(mode=False):
    """
    collect dataframe and all relevant paths:
    """
    # get working directory path
    path = os.getcwd()

    add = "../" if path[-3:] == "src" else ""

    name = 'datainfo'
    pic_folder = 'PNG'
    seg_folder = 'SEG'
    mask_folder = 'MASK'

    if mode:
        name = f'{name}_external'
        pic_folder = f'{pic_folder}_external'
        seg_folder = f'{seg_folder}_external'

    name = f'{name}.xlsx' if mode else f'{name}.csv'

    # get all releevant paths
    paths_loc = {
        "csv": os.path.join(path, f'{add}{name}'),
        "pic": os.path.join(path, f'{add}{pic_folder}'),
        "seg": os.path.join(path, f'{add}{seg_folder}'),
        "mask": os.path.join(path, f'{add}{mask_folder}'),
        "nrrd": os.path.join(path, f'{add}/radiomics/image')
    }

    # get data_fr
    if mode:
        data_frame = pd.read_excel(paths_loc["csv"])
    else:
        data_frame = pd.read_csv(
            paths_loc["csv"], header='infer', delimiter=';')

    return data_frame, paths_loc


def regenerate_ex_names(paths_loc, new_path='../PNG_external'):
    """redefine the external path names"""
    # append idlist
    data_fr_external = pd.read_excel(paths_loc["csv"])
    idlist = np.array(list(range(1, len(data_fr_external)+1)))
    np.random.shuffle(idlist)
    data_fr_external['id'] = idlist
    data_fr_external.to_excel(paths_loc["csv"])

    data_fr_external = pd.read_excel(paths_loc["csv"])
    old_path = paths_loc['pic']
    for num, fname in zip(data_fr_external['id'], data_fr_external[F_KEY]):
        filename = f'{old_path}/{fname}.png'
        img = Image.open(filename)
        filename_new = f'{new_path}/{num}.png'
        img.save(filename_new)


# %% Perform dataset preparation
if __name__ == '__main__':
    SIMPLE = True

    for external_mode in [False, True]:
        # get the paths
        data_fr, paths = get_data_fr_paths(mode=external_mode)

        # %% show the distributions
        print('\n\nDataset information:\n')
        ages = get_data_fr_dis(data_fr, mode=external_mode)

        # %% add the detailed classes to the dataframe
        print('\n\nAdd the detailed classes to the csv')
        add_classes_to_csv(paths["csv"], mode=external_mode)

        # %% build the coco-formated json
        print('\n\nTransform to coco format')
        cocos = get_cocos_from_data_fr(
            data_fr, paths, save=True, simple=SIMPLE, newmode=0, ex_mode=external_mode)


# %%
