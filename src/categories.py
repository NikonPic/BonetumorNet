# %%
def make_cat(simple=True):
    """fill the categories manually"""
    if simple:
        cat_list = [
            {
                "id": 0,
                "name": "Benign Tumor",
            },
            {
                "id": 1,
                "name": "Malignant Tumor",
            }
        ]
        cat_mapping = [0, 1]
        return cat_list, cat_mapping

    cat_list = [
        {
            "supercategory": "Malignant",
            "id": 0,
            "name": "Osteosarcoma",
        },
        {
            "supercategory": "Malignant",
            "id": 1,
            "name": "Chondrosarcoma",
        },
        {
            "supercategory": "Malignant",
            "id": 2,
            "name": "Ewing’s sarcoma",
        },
        {
            "supercategory": "Benign",
            "id": 3,
            "name": "Enchondroma",
        },
        {
            "supercategory": "Benign",
            "id": 4,
            "name": "NOF",
        },
    ]
    cat_mapping = {
        # The names from datainfo are used here!
        "Osteosarkom": 0,
        "Chondrosarkom": 1,
        "Ewing-Sarkom": 2,
        "Enchondrom": 3,
        "NOF": 4,
    }
    return cat_list, cat_mapping

def get_cat_list(simple):
    if simple:
        cat_list = [
            {
                "id": 0,
                "name": "Benign Tumor",
            },
            {
                "id": 1,
                "name": "Malignant Tumor",
            }
        ]
    else:
        cat_list = [
        # malignant first
        {
            "supercategory": "Malignant",
            "id": 1,
            "name": "Chondrosarcoma",
        },
        {
            "supercategory": "Malignant",
            "id": 2,
            "name": "Osteosarcoma",
        },
        {
            "supercategory": "Malignant",
            "id": 3,
            "name": "Ewing sarcoma",
        },
        {
            "supercategory": "Malignant",
            "id": 4,
            "name": "Plasma cell myeloma",
        },
        {
            "supercategory": "Malignant",
            "id": 5,
            "name": "NHL B Cell",
        },
        # now benign
        {
            "supercategory": "Benign",
            "id": 6,
            "name": "Osteochondroma",
        },
        {
            "supercategory": "Benign",
            "id": 7,
            "name": "Enchondroma",
        },
        {
            "supercategory": "Benign",
            "id": 8,
            "name": "Chondroblastoma",
        },
        {
            "supercategory": "Benign",
            "id": 9,
            "name": "Osteoid osteoma",
        },
        {
            "supercategory": "Benign",
            "id": 10,
            "name": "Non-ossifying fibroma",
        },
        {
            "supercategory": "Benign",
            "id": 11,
            "name": "Giant cell tumour of bone",
        },
        {
            "supercategory": "Benign",
            "id": 12,
            "name": "Chordoma",
        },
        {
            "supercategory": "Benign",
            "id": 13,
            "name": "Haemangioma",
        },
        {
            "supercategory": "Benign",
            "id": 14,
            "name": "Aneurysmal bone cyst",
        },
        {
            "supercategory": "Benign",
            "id": 15,
            "name": "Simple bone cyst",
        },
        {
            "supercategory": "Benign",
            "id": 16,
            "name": "Fibrous dysplasia",
        },
    ]
    return cat_list

def make_cat_advanced(simple=True, yolo=False):
    """fill the categories manually"""
    cat_list = get_cat_list(simple)
    if simple:
        if yolo:
            cat_mapping = {
                "benign" : 0,
                "malign": 1,
            }
        else:
            cat_mapping = [0, 1]

        return cat_list, cat_mapping
    # The names from datainfo are used here!
    cat_mapping = {
        # malign
        "Chondrosarkom": 1,
        "Osteosarkom": 2,
        "Ewing-Sarkom": 3,
        "Plasmozytom / Multiples Myelom": 4,
        "NHL vom B-Zell-Typ": 5,
        # benign
        "Osteochondrom": 6,
        "Enchondrom": 7,
        "Chondroblastom": 8,
        "Osteoidosteom": 9,
        "NOF": 10,
        "Riesenzelltumor": 11,
        "Chordom": 12,
        "Hämangiom": 13,
        "Knochenzyste, aneurysmatische": 14,
        "Knochenzyste, solitär": 15,
        "Dysplasie, fibröse": 16,
    }
    return cat_list, cat_mapping


cat_mapping_new = {
    # lis: entity, ben/mal, ben/lok/mal
    # malign
    "Chondrosarkom": [0, 1, 2, 3, 2, 1, 0],
    "Osteosarkom": [1, 1, 2, 3, 2, 1, 1],
    "Ewing-Sarkom": [2, 1, 2, 3, 2, 1, 1],
    "Plasmozytom / Multiples Myelom": [3, 1, 2, 3, 2, 1, 2],
    "NHL vom B-Zell-Typ": [4, 1, 2, 3, 2, 1, 2],
    # benign
    "Osteochondrom": [5, 0, 0, 2, 2, 0, 0],
    "Enchondrom": [6, 0, 0, 1, 1, 0, 0],
    "Chondroblastom": [7, 1, 1, 2, 2, 0, 0],
    "Osteoidosteom": [8, 0, 0, 2, 2, 0, 1],
    "NOF": [9, 0, 0, 0, 0, 0, 3],
    "Riesenzelltumor": [10, 1, 1, 3, 2, 0, 3],
    "Chordom": [11, 1, 1, 2, 2, 0, 3],
    "Hämangiom": [12, 0, 0, 2, 2, 0, 3],
    "Knochenzyste, aneurysmatische": [13, 0, 0, 2, 2, 0, 3],
    "Knochenzyste, solitär": [14, 0, 0, 2, 2, 0, 3],
    "Dysplasie, fibröse": [15, 0, 0, 2, 2, 0, 3],
}



reverse_cat_list = [
    "Chondrosarkom",
    "Osteosarkom",
    "Ewing-Sarkom",
    "Plasmozytom / Multiples Myelom",
    "NHL vom B-Zell-Typ",
    "Osteochondrom",
    "Enchondrom",
    "Chondroblastom",
    "Osteoidosteom",
    "NOF",
    "Riesenzelltumor",
    "Chordom",
    "Hämangiom",
    "Knochenzyste, aneurysmatische",
    "Knochenzyste, solitär",
    "Dysplasie, fibröse",
]



cat_naming_new = [
    {
        'name' : 'All Entities',
        'cat' : 16,
        'index' : 0,
        'catnames': reverse_cat_list,
    },
    {
        'name' : 'Aggressive - Non Aggressive',
        'cat' : 2,
        'index' : 1,
        'catnames': [
            'Non Aggressive',
            'Aggresive'
        ],
    },
    {
        'name' : 'Benigne - Local Aggressive - Maligne',
        'cat' : 3,
        'index' : 2,
        'catnames' : [
            'Benigne',
            'Local Aggressive',
            'Maligne'
        ]
    },
    {
        'name' : 'Grade of clinical workflow',
        'cat' : 4,
        'index' : 3,
        'catnames' : [
            '0',
            '1',
            '2',
            '3'
        ]
    },
    {
        'name' : 'Grade for clinical workflow (2 + 3 = 2 > assessment in MSK center needed)',
        'cat' : 3,
        'index' : 4,
        'catnames' : [
            '0',
            '1',
            '2 + 3',
        ]
    },
    {
        'name' : 'malignant',
        'cat' : 2,
        'index' : 5,
        'catnames' : [
            'Benign',
            'Malign'
        ]
    },
    {
        'name' : 'Super Entity (chon: 0, osteo:1, meta:2, other:3)',
        'cat' : 4,
        'index' : 6,
        'catnames' : [
            'chondrogenic tumor',
            'osteogenic tumor',
            'meta /haema',
            'other entities'
        ]
    },
]


malign_int = [
    0, 1, 2, 3, 4, 11
]

benign_int = [
    5, 6, 7, 8, 9, 10, 12, 13, 14, 15
]

# %%

