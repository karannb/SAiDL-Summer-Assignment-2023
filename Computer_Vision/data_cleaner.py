#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 10:00:26 2023

@author: karan_bania
"""

import pandas as pd
import os

l_temp=os.listdir("PhraseCutDataset/data/VGPhraseCut_v0/images_train")
l_train=[x.split('.')[0] for x in l_temp]
l_train_int=[eval(i) for i in l_train]

l_temp=os.listdir("PhraseCutDataset/data/VGPhraseCut_v0/images_val")
l_val=[x.split('.')[0] for x in l_temp]
l_val_int=[eval(i) for i in l_val]


#TRAIN - 
df = pd.read_json('./PhraseCutDataset/data/VgPhraseCut_v0/refer_train.json')
df.drop(['phrase_structure','task_id','Polygons','instance_boxes','ann_ids','phrase'], axis = 1, inplace = True)

#df.set_index('task_id')

#columns_titles = ["task_id","image_id", "ann_ids", "phrase", "instance_boxes", "Polygons"]
#df=df.reindex(columns=columns_titles)

#df['task_id'] = df['task_id'].apply(lambda x: "'" + str(x) + "'")
#df['phrase'] = df['phrase'].apply(lambda x: "'" + str(x) + "'")

df = df[df['image_id'].isin(l_train_int)]

df.to_csv('data_train.csv', index = False)

#VALIDATION - 
df = pd.read_json('./PhraseCutDataset/data/VgPhraseCut_v0/refer_val.json')
df.drop(['phrase_structure','task_id','Polygons','instance_boxes','ann_ids','phrase'], axis = 1, inplace = True)

#df.set_index('task_id')

#columns_titles = ["task_id","image_id", "ann_ids", "phrase", "instance_boxes", "Polygons"]
#df=df.reindex(columns=columns_titles)

#df['task_id'] = df['task_id'].apply(lambda x: "'" + str(x) + "'")
#df['phrase'] = df['phrase'].apply(lambda x: "'" + str(x) + "'")

df = df[df['image_id'].isin(l_val_int)]

df.to_csv('data_val.csv', index = False)