#!/bin/bash


time CUDA_VISIBLE_DEVICES=0 python src/parser.py --outdir ./output --train coco_train.conll --dev coco_dev.conll --epochs 30 --lstmdims 256 --lstmlayers 2  --k 3 --usehead --userl 


#time CUDA_VISIBLE_DEVICES=0 python src/parser.py  --predict --outdir /media/Work_HD/yswang/bistparser/barchybrid_v1_0/output --test /media/Work_HD/yswang/dataset/glove/glove.6B/dev_current_random.conll --model /media/Work_HD/yswang/bist-parser/barchybrid_v1_0/output/barchybrid.model1.tmp --params /media/Work_HD/yswang/bist-parser/barchybrid_v1_0/output/params.pickle
