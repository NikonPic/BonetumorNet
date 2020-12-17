# %%
import os
import pandas as pd
import detectron2
from categories import make_categories_advanced, make_categories, cat_mapping_new, cat_naming_new, reverse_cat_list, malign_int, benign_int
from sklearn.metrics import confusion_matrix
from utils_detectron import plot_confusion_matrix
import numpy as np


f_eval_doc = './evalDoctors'
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
sel_id = 0


def extract_local_info(loc_file, f_eval_doc):
    """
    extract the information from filename and the txt data
    """
    file_id = int(loc_file.split('_')[3].split('.')[0])
    split_info = open(f'{f_eval_doc}/{loc_file}', 'r').read().split('///')
    entity = split_info[2]
    workflow = split_info[3]
    
    try:
        workflow = int(workflow[10])
    except:
        workflow = 2

    if 'Dysplasie' in entity:
        entity = 'Dysplasie, fibröse'

    if 'mangiom' in entity:
        entity = 'Hämangiom'
    
    if 'Knochenzyste, sol' in entity:
        entity = 'Knochenzyste, solitär'

    return file_id, entity, workflow

def print_confinfo(conf):
    """print all relevant infos for confidence intervalls"""
    tp = conf[1, 1]
    tn = conf[0, 0]
    fp = conf[0, 1] + conf[0, 2]
    fn = conf[1, 0] + conf[1, 2]

    sens = round(tp / (tp + fn), 2)
    sens_high, sens_low = get_ci(sens, n=tp+fn, printit=False)
    spec = round(tn / (tn + fp), 2)
    spec_high, spec_low = get_ci(spec, n=tn+fp, printit=False)
    acc  = round((tn + tp) / (tn + fp + tp + fn), 3)
    acc_high, acc_low = get_ci(acc, n=tn + fp + tp + fn, printit=False, digits=3)


    print(f'sensitivity : {sens} ({tp} of { (tp + fn)}), 95% CI: {sens_high}% {sens_low}%')
    print(f'specificity : {spec} ({tn} of { (tn + fp)}), 95% CI: {spec_high}% {spec_low}%')
    print(f'accuracy : {acc} ({tn + tp} of { (tn + fp + tp + fn)}), 95% CI: {acc_high}% {acc_low}%')

def get_ci(acc, n=140, const=1.96, digits=4, printit=True):
    """calculate confidence intervall"""
    acc = round(acc, digits)
    error = 1 - acc
    ci_low  = round((1 - (error - const * np.sqrt((error * (1 - error)) / n))), digits) * 100
    ci_high = round((1 - (error + const * np.sqrt((error * (1 - error)) / n))), digits) * 100
    if printit:
        print(f'Acc: {acc*100}%, 95% CI: {ci_high}%, {ci_low}%)')
    return ci_high, ci_low

# %%

df = pd.read_excel(f_xlsx)


# %%
# get all results of ALex / claudio
filelist = os.listdir(f_eval_doc)
files_person = [file for file in filelist  if (name_d[sel_id] in file)]

ids = []
entities = []
workflows = []

for loc_file in files_person:
    file_id, entity, workflow = extract_local_info(loc_file, f_eval_doc)

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
        score3 +=1
    
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

entity_conf   = confusion_matrix(ent_true, ent_pred)
workflow_conf = confusion_matrix(workflow_true, workflow_pred) 
benmal_conf   = confusion_matrix(benmal_true, benmal_pred)

plot_confusion_matrix(entity_conf, reverse_cat_list)
plot_confusion_matrix(workflow_conf, wf_title)
plot_confusion_matrix(benmal_conf, benmal_title)


print_confinfo(benmal_conf)


print(f'Entities: {round(100*score / 111, 3)}%')
print(f'MalBen: {round(100*score2 / 111, 2)}%')
print(f'Workflow: {round(100*score3 / 111, 2)}%')
get_ci(score / 111, n=111, digits=4)

# %%
