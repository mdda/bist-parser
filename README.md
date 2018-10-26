Scene Graph Parsing as Dependency Parsing
===================

This repository contains code for Scene Graph Parsing as Dependency Parsing, NAACL 2018.

If you use the code, please cite 
``` 
@inproceedings{wang2018sgparser,     
  title={Scene Graph Parsing as Dependency Parsing},  
  author={Wang, Yu-Siang and Liu, Chenxi and Zeng, Xiaohui and Yuille, Alan},  
  booktitle={NAACL},  
  year={2018}
} 
```


## Requirements
- python 2.7
- PyTorch v0.3

## Files
./model
- include training codes

./preprocess
- include mapping from json data to conll file 

./coco-caption
- evaluate coco caption with dependency sg parser

## Customized Dependency Scene Graph Parser
#### Data Split
For training (development) data, we adopt the overlap images between Visual Genome training (development) dataset and MSCOCO training (development) dataset.

#### Preprocessing ####

TODO : Update the instructions, since parsing streaming json is *way* more efficient than manipulating these
files in 10 sections, and stitching them back together...

1. BETTER : 
```
cat image_data.json    | jq -n --compact-output --stream 'fromstream(1|truncate_stream(inputs))' > image_data.json.rows
cat attributes.json    | jq -n --compact-output --stream 'fromstream(1|truncate_stream(inputs))' > attributes.json.rows
cat region_graphs.json | jq -n --compact-output --stream 'fromstream(1|truncate_stream(inputs))' > region_graphs.json.rows
```

<strike>
1. OLD : Split the Visual Genome {image_data.json, region_graphs.json, attributes.json} files each into 10 pieces, and name these files ```x_%num.json```, 
   where x={image_data, all_region_graphs, all_attributes} and num={0..9}. 
   (The size for the every first 9 pieces is all_region_graph.size()/10) and put them into ```preprocess/data```
   The reason to split to 10 pieces is for acclerating the preprocessing speed. 
</strike>  

1. Move the ```.json.rows``` files into ```./preprocess/data/``` (or symlink the ```./preprocess/data``` directory to the folder with the downloaded / preprocessed data).

2. CHECK : Set nltk wordnet path in line 10 in data_to_conll.py



run
```
bash ./preprocess.sh
```
then run :

<strike>
```
python split_preprocess.py NUM TARGET
```

where NUM={0..9}, TARGET={coco_train, coco_dev}, which means there are total 20 commands to run.
We strongly suggest you to open multiple terminals to execute above commands to fasten the preprocesseing time.
</strike>

The following creates the files ```./intermediate/pre_coco_TARGET.json.rows``` :
```
python split_preprocess.py 0000 coco_train
python split_preprocess.py 0000 coco_dev

#NOPE python split.py merge coco_train 
#NOPE python split.py merge coco_dev
```
where TARGET={coco_train, coco_dev}, which means there are 2 commands to run.

Finally, run the following to convert (via the paper's Algorithm 1) ```./intermediate/pre_coco_XXX.json.rows``` 
to ```./output/coco_XXX.conll``` :

```
python data_to_conll.py --input pre_coco_train.json --output coco_train.conll --train True
python data_to_conll.py --input pre_coco_dev.json   --output coco_dev.conll   --train False
```

--input:  processed data generated by split.py
--output: desired output
--train : determine whether the file is used for training or not



#### Training
```
cd ./model/
bash run.sh
```
In run.sh, you can set the paramters
```
time CUDA_VISIBLE_DEVICES=2 python src/parser.py --outdir ./output --train ./output/coco_train.conll --dev /output/coco_dev.conll \
                                                 --epochs 30 --lstmdims 256 --lstmlayers 2  --k 3 --usehead --userl
```
Please ignore .txt files which are useless

#### Validation

```
cd ./model/src/utils/evaluation_script/
bash eval.sh
```
In eval.sh, pre_coco_dev.json is the ground truth file.

```
time python spice_eval.py pre_coco_dev.json output/predict_coco.conll
```
Remember to set nltk_wordnet path in `./model/src/utils/evaluation_script/spice_eval.py` to use wordnet for spice metric


#### Testing

```
python src/parser.py --predict --outdir [results directory] --test test.conll [--extrn extrn.vectors] --model [trained model file] --params [param file generate during training]
```













