import json
import codecs
import random
import pickle
from sklearn.utils import shuffle
import sys

#NUM    = sys.argv[1]

ACTION = sys.argv[1]
TARGET = sys.argv[2]

#def read_region_graphs():
# return json.load(open("./data/all_region_graphs_%d.json" %NUM, "r"))


def overlap_NOTUSED():
  #t1 = json.load(open("pre_random_train.json", 'r'))
  #t2 = json.load(open("pre_random_dev.json", 'r'))

  t1 = json.load(open("pre_coco_train.json", 'r'))
  t2 = json.load(open("pre_coco_dev.json", 'r'))

  s1 = set()
  s2 = set()
  s1_idx = dict()
  s2_idx = dict()
  sl1 = []
  sl2 = []

  for tt1 in t1:
    s1.add(tt1[0][0])
    #if tt1[0][0] not in sl1:
    # s1.append(tt1)
    # sl1.append(tt1[0][0])
  for tt2 in t2:
    s2.add(tt2[0][0])
    #if tt2[0][0] not in sl2:
    # s2.append(tt2)
    # sl2.append(tt2[0][0])
  for s1_ele in s1:
    s1_idx[s1_ele] = []
  for s2_ele in s2:
    s2_idx[s2_ele] = []

  for tt1 in t1:
    s1_idx[tt1[0][0]].append(tt1[1:])

  for tt2 in t2:
    s2_idx[tt2[0][0]].append(tt2[1:])

  inter = s1.intersection(s2)

  total = len(inter)
  count_iter = 0
  count = 0
  for inter_sent in inter:
    print("No:%d/%d" % (count_iter, total))
    temp_s1 = s1_idx[inter_sent]
    temp_s2 = s2_idx[inter_sent]

    for ss1 in temp_s1:
      for ss2 in temp_s2:
        if ss1 == ss2:
          count += 1
          break
    count_iter+= 1

  print count
  print len(t2)

  exit()

  count = 0

  for ss1_id, ss1 in enumerate(sl1):
    for ss2_id, ss2 in enumerate(sl2):
      if sl1[ss1_id] == sl2[ss2_id]:
      #if s1[ss1_id] == s2[ss2_id]:
        count += 1
        break

  print count
  

def process_labels_orig():
  all_region_graphs = []
  image_data = []
  for data_id in range(10):
    part_region = json.load(open("./data/all_region_graphs_%d.json" % data_id, "r"))
    all_region_graphs.extend(part_region)

    part_image_data = json.load(open('./data/image_data_%d.json' % data_id,'r'))
    image_data.extend(part_image_data)

  #print "Process %d/10" % NUM
  #all_region_graphs = read_region_graphs()
  #image_data = json.load(open('image_data_%d.json' %NUM,'r'))
  
  image_iter_id = open('image_iter_id.txt', 'w')
  image_ids = open('image_id.txt', 'w')
  image_coco_ids = open('image_inter_coco_id.txt', 'w')
  total_images = len(all_region_graphs)

  #assert len(all_region_graphs) == len(image_data)
  print("Total images: %d" % total_images)
  for im in range(len(all_region_graphs)):
    print("Progress: images:  %d/%d" % (im, total_images))
    #for k, v in all_region_graphs[im].iteritems():
    # print k
    
    #print all_region_graphs[im]['']

    if image_data[im]['coco_id'] == None:
      continue
    else:
      image_ids.write(str(all_region_graphs[im]['image_id'])+'\n')
      image_iter_id.write(str(im)+'\n')
      image_coco_ids.write(str(image_data[im]['coco_id'])+'\n')


def process_labels_rowwise():
  # Since all that was happening in this function was row-wise, no need to read everything at once...
  
  # This is just for show...  So find the length of the image_data file...
  with open('./data/image_data.json.rows', 'r') as f:
    total_image_count = len( f.readlines() )
  #assert len(all_region_graphs) == len(image_data)
  print("Total images: %d" % total_image_count)
  
  # Saving the data to these files...
  image_iter_id  = open('image_iter_id.txt', 'w')
  image_ids      = open('image_id.txt', 'w')
  image_coco_ids = open('image_inter_coco_id.txt', 'w')

  with open('./data/image_data.json.rows', 'r') as im_file, open('./data/region_graphs.json.rows', 'r')  as reg_file:
    for im, (image_data_json, region_graph_json) in enumerate(zip(im_file, reg_file)):
      if im % 100==0:
        print("Progress: images:  %d/%d" % (im, total_image_count))
        
      image_data    = json.loads(image_data_json)
      region_graphs = json.loads(region_graph_json)
      
      #for k, v in region_graphs.iteritems():
      #  print(k)
      
      #print(region_graphs[''])

      if image_data['coco_id'] == None:
        continue
      else:
        image_ids.write(str(region_graphs['image_id'])+'\n')
        image_iter_id.write(str(im)+'\n')
        image_coco_ids.write(str(image_data['coco_id'])+'\n')


def generate_coco_split():
  train_id_path = 'train_id.p' 
  dev_id_path   = 'dev_id.p'

  train_id =  pickle.load(open(train_id_path, 'r'))
  dev_id   =  pickle.load(open(dev_id_path, 'r'))

  train_vg_sent = []
  dev_vg_sent   = []

  f_image_id = open('image_iter_id.txt', 'r')
  coco_image_id = open('image_inter_coco_id.txt', 'r')
  img_ids = []
  coco_img_ids = []

  #for line in f_image_id.readlines():
  for line in f_image_id:
    line = line.strip()
    img_ids.append(line)

  #for line in coco_image_id.readlines():
  for line in coco_image_id:
    line = line.strip()
    coco_img_ids.append(line)

  f_image_id = img_ids

  for img_id_idx, img_id in enumerate(coco_img_ids):
    if int(img_id) in train_id:
      train_vg_sent.append(f_image_id[img_id_idx])
    elif int(img_id) in dev_id:
      dev_vg_sent.append(f_image_id[img_id_idx])

  json.dump(train_vg_sent, open("coco_train_id.json", "w"))
  json.dump(dev_vg_sent,   open("coco_dev_id.json", "w"))
  

def generate_random_split():
  f_image_id = open('image_iter_id.txt', 'r')
  img_ids = []
  for line in f_image_id.readlines():
    line = line.strip()
    img_ids.append(line)
  f_image_id = img_ids
  random.seed(1)
  random.shuffle(f_image_id)
  batch = len(f_image_id)/10
  random_train_id = f_image_id[:6*batch]
  random_dev_id   = f_image_id[6*batch: 8*batch]
  random_test_id  = f_image_id[8*batch:]

  json.dump(random_train_id, open("random_train_id.json", "w"))
  json.dump(random_dev_id,   open("random_dev_id.json", "w"))
  json.dump(random_test_id,  open("random_test_id.json", "w"))


def process_ids_orig(target):
  data_path = target+'_id.json'
  id_list = json.load(open(data_path, 'r'))

  all_region_graphs = []
  all_attributes    = []
  for data_id in range(10):
    part_region = json.load(open("./data/all_region_graphs_%d.json" % data_id, "r"))
    all_region_graphs.extend(part_region)

    part_attr = json.load(open("./data/all_attributes_%d.json" % data_id, "r"))
    all_attributes.extend(part_attr)

  temp_all_attributes = []
  for img_id in id_list:
    temp_all_attributes.append(all_attributes[int(img_id)])

  temp_all_region_graphs = []
  for img_id in id_list:
    temp_all_region_graphs.append(all_region_graphs[int(img_id)])

  batch = len(temp_all_region_graphs)/10
  for i in range(10):
    if i != 9:
      json.dump(temp_all_region_graphs[i*batch: (i+1)*batch], open(target+"_region_%d.json"%i, 'w'))
      json.dump(temp_all_attributes[i*batch: (i+1)*batch], open(target+"_attr_%d.json"%i, 'w'))

    else:
      json.dump(temp_all_region_graphs[i*batch:], open(target+"_region_%d.json"%i, 'w'))
      json.dump(temp_all_attributes[i*batch:], open(target+"_attr_%d.json"%i, 'w'))     


def process_ids_rowwise(target):
  data_path = target+'_id.json'
  id_list = json.load(open(data_path, 'r'))
  print( id_list )
  exit(0)

  #all_region_graphs = []
  #all_attributes    = []

  temp_all_attributes = []
  temp_all_region_graphs = []

  for img_id in id_list:
    temp_all_attributes.append(all_attributes[int(img_id)])
    temp_all_region_graphs.append(all_region_graphs[int(img_id)])

  batch = len(temp_all_region_graphs)/10
  for i in range(10):
    if i != 9:
      json.dump(temp_all_region_graphs[i*batch: (i+1)*batch], open(target+"_region_%d.json"%i, 'w'))
      json.dump(temp_all_attributes[i*batch: (i+1)*batch], open(target+"_attr_%d.json"%i, 'w'))

    else:
      json.dump(temp_all_region_graphs[i*batch:], open(target+"_region_%d.json"%i, 'w'))
      json.dump(temp_all_attributes[i*batch:], open(target+"_attr_%d.json"%i, 'w'))     




def merge_data(target):
  full_data = []
  for i in range(10):
    temp_data = json.load(open("pre_"+target+"_%d.json"%i, 'r'))
    full_data.extend(temp_data)
  json.dump(full_data, open("pre_"+target+".json", "w"))



#target = "random_train"
#target = "random_dev"
#target = "random_test"
#target = "coco_train"
target = TARGET

if ACTION != 'merge':
  #process_labels_orig()
  process_labels_rowwise()
  
  generate_coco_split()
  #generate_random_split()
  
  #process_ids_orig(target)
  process_ids_rowwise(target)
  
else:
  merge_data(target)
