#import numpy as np
#import pickle
#import json
#import utils as ut
#import re

# input json: 
# 0: phrase
# 1: object
# 2: attributes with their objects
# 3: relations

# train data
# 0: none
# 1: objects
# 2: attributes
# 3: relations


def label_data(val_pre):
    objects = val_pre[1]
    attributes = val_pre[2]
    relations = val_pre[3]

    all_rel  = []
    all_attr_pair = []
    all_rel_pair  = []
    all_pair = []

    for i in range(len(objects)):
      all_pair.append([(objects[i]).lower()])

    for i in range(len(attributes)): 
      for attr in attributes[i][1]:
        all_pair.append(tuple([attributes[i][0].lower(),attr.lower()]))
        
    for i in range(len(relations)):
      all_pair.append(tuple([relations[i][0].lower(), relations[i][1].lower(), relations[i][2].lower()]))

    #print(all_pair)
    return all_pair
    
#label_train_data()
