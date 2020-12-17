# %%
import os
import pandas as pd
from categories import cat_mapping_new, reverse_cat_list, malign_int, benign_int
from sklearn.metrics import confusion_matrix
from utils_detectron import plot_confusion_matrix
from detec_helper import get_ci
import numpy as np


file_eval_doc = './evalDoctors'
f_xlsx = 'datainfo_external.xlsx'

df_ent = 'Tumor.Entitaet'
df_id = 'id'
df_workflow = 'Grade for clinical workflow (2 + 3 = 2 > assessment in MSK center needed)'

wf_title = [
    "workflow 0",
    "workflow 1",
    "workflow 2"
]
benmal_title = ["Benign", "Malignant", "Cannot be classified"]

name_d = ['1', '2']
SEL_ID = 0


def extract_local_info(loc_file):
    """
    extract the information from filename and the txt data
    """
    loc_file_id = int(loc_file.split('_')[3].split('.')[0])
    split_info = open(f'{file_eval_doc}/{loc_file}', 'r').read().split('///')
    loc_entity = split_info[2]
    workflow = split_info[3]
    workflow = int(workflow[10])

    if 'Dysplasie' in loc_entity:
        loc_entity = 'Dysplasie, fibröse'
    if 'mangiom' in loc_entity:
        loc_entity = 'Hämangiom'
    if 'Knochenzyste, sol' in loc_entity:
        loc_entity = 'Knochenzyste, solitär'

    return loc_file_id, loc_entity, workflow


def print_confinfo(conf):
    """print all relevant infos for confidence intervalls"""
    true_pos = conf[1, 1]
    true_neg = conf[0, 0]
    false_pos = conf[0, 1] + conf[0, 2]
    false_neg = conf[1, 0] + conf[1, 2]

    sens = round(true_pos / (true_pos + false_neg), 2)
    sens_high, sens_low = get_ci(sens, n=true_pos+false_neg, printit=False)
    spec = round(true_neg / (true_neg + false_pos), 2)
    spec_high, spec_low = get_ci(spec, n=true_neg+false_pos, printit=False)
    acc = round((true_neg + true_pos) / (true_neg +
                                         false_pos + true_pos + false_neg), 3)
    acc_high, acc_low = get_ci(
        acc, n=true_neg + false_pos + true_pos + false_neg, printit=False, digits=3)

    print(
        f'sensitivity : {sens} ({true_pos} of { (true_pos + false_neg)}), 95% CI: {sens_high}% {sens_low}%')
    print(
        f'specificity : {spec} ({true_neg} of { (true_neg + false_pos)}), 95% CI: {spec_high}% {spec_low}%')
    print(f'accuracy : {acc} ({true_neg + true_pos} of { (true_neg + false_pos + true_pos + false_neg)}), 95% CI: {acc_high}% {acc_low}%')

# %% read the dataframe

df = pd.read_excel(f_xlsx)


# %%
# get all results of ALex / claudio
filelist = os.listdir(file_eval_doc)
files_person = [file for file in filelist if name_d[SEL_ID] in file]

ids = []
entities = []
workflows = []

for loc_file in files_person:
    file_id, entity, workflow = extract_local_info(loc_file)

    ids.append(file_id)
    entities.append(entity)
    workflows.append(workflow)

# %%
score = 0
score2 = 0
score3 = 0

ent_pred = []
ent_true = []

benmal_pred = []
benmal_true = []

workflow_pred = []
workflow_true = []

for true_id, true_entity, true_workflow in zip(df[df_id], df[df_ent], df[df_workflow]):
    # first match id
    selected_row = ids.index(true_id)
    entity = entities[selected_row]
    workflow = workflows[selected_row]

    if entity == true_entity:
        score += 1
    if entity in cat_mapping_new.keys():
        pred_int = cat_mapping_new[entity][0]
    else:
        pred_int = 20
    true_int = cat_mapping_new[true_entity][0]

    if (pred_int in malign_int) and (true_int in malign_int):
        score2 += 1
    if (pred_int in benign_int) and (true_int in benign_int):
        score2 += 1
    if workflow == true_workflow:
        score3 += 1

    ent_pred.append(entity)
    ent_true.append(true_entity)

    workflow_pred.append(workflow)
    workflow_true.append(true_workflow)

    pred_benmal = 2
    if (pred_int in benign_int):
        pred_benmal = 0

    if (pred_int in malign_int):
        pred_benmal = 1

    true_benmal = 0
    if (true_int in malign_int):
        true_benmal = 1

    benmal_pred.append(pred_benmal)
    benmal_true.append(true_benmal)


if len(reverse_cat_list) < 17:
    reverse_cat_list.append('Undefined')

entity_conf = confusion_matrix(ent_true, ent_pred)
workflow_conf = confusion_matrix(workflow_true, workflow_pred)
benmal_conf = confusion_matrix(benmal_true, benmal_pred)

plot_confusion_matrix(entity_conf, reverse_cat_list)
plot_confusion_matrix(workflow_conf, wf_title)
plot_confusion_matrix(benmal_conf, benmal_title)

print_confinfo(benmal_conf)

print(f'Entities: {round(100*score / 111, 3)}%')
print(f'MalBen: {round(100*score2 / 111, 2)}%')
print(f'Workflow: {round(100*score3 / 111, 2)}%')
get_ci(score / 111, n=111, digits=4)
