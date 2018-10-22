import pickle
import cPickle as cp
import json
import codecs
import sys

NUM    = int(sys.argv[1])  #range: 0~9
TARGET = sys.argv[2]  #coco_train or coco_dev



def read_attributes():
  #all_attributes = []

  ### READ Random Split
  #id_list = json.load(open('random_train_id.json', 'r'))
  #id_list   = json.load(open('random_dev_id.json', 'r'))
  #id_list  = json.load(open('random_test_id.json', 'r'))

  ## CAPTION
  #id_list = json.load(open('coco_train_id.json', 'r'))
  #id_list = json.load(open('coco_dev_id.json', 'r'))

  #for data_id in range(10):
  # part_attr = json.load(open("./all_attributes_%d.json" % data_id, "r"))
  # all_attributes.extend(part_attr)

  #temp_all_attributes = []
  #for img_id in id_list:
  # temp_all_attributes.append(all_attributes[int(img_id)])

  #all_attributes = temp_all_attributes

  return json.load(open(TARGET+"_attr_%d.json"%NUM, 'r'))

def read_region_graphs():
  #all_region_graphs = []
  #image_data = []

  ### READ Random Split
  #id_list = json.load(open('random_train_id.json', 'r'))
  #id_list   = json.load(open('random_dev_id.json', 'r'))
  #id_list  = json.load(open('random_test_id.json', 'r'))

  ## CAPTION
  #id_list = json.load(open('coco_train_id.json', 'r'))
  #id_list = json.load(open('coco_dev_id.json', 'r'))


  #for data_id in range(10):
  # part_region = json.load(open("./all_region_graphs_%d.json" % data_id, "r"))
  # all_region_graphs.extend(part_region)

  # part_image_data = json.load(open('image_data_%d.json' % data_id,'r'))
  # image_data.extend(part_image_data)

  #temp_all_region_graphs = []
  #temp_image_data        = []
  #for img_id in id_list:
  # temp_all_region_graphs.append(all_region_graphs[int(img_id)])
  # temp_image_data.append(image_data[int(img_id)])

  #all_region_graphs = temp_all_region_graphs
  #image_data        = temp_image_data

  return json.load(open(TARGET+'_region_%d.json'%NUM, 'r'))


def process_labels():
  all_region_graphs = read_region_graphs()
  all_attributes   = read_attributes()

  all_labels = []
  
  total_region_graphs = []
  total_images = len(all_region_graphs)
  print("Total images: %d" % total_images)
  
  assert len(all_region_graphs) == len(all_attributes)
  #assert len(all_region_graphs) == len(image_data)

  for im in range(len(all_region_graphs)):
    
    regions = all_region_graphs[im]['regions']
    im_attributes = all_attributes[im]['attributes']
    obj_to_attr = dict()
    
    for attr in range(len(im_attributes)):
      obj_id = im_attributes[attr]['object_id']
      try:
        obj_to_attr[obj_id] = im_attributes[attr]['attributes']
      except:
        continue

    total_regions = len(regions)
    count_region = 0
    for region in regions:
      if len(total_region_graphs) != 0:
        if len(region['phrase'].strip().split()) == 0:
          continue
        if total_region_graphs[-1][0][0] == region['phrase']:
          continue

      count_region += 1
      print("Progress: images:  %d/%d, regions:  %d/%d" % (im, total_images, count_region, total_regions))

      region_objects = []
      region_attributes = []
      region_relaions = []
      region_phrase = []
      obj_id_to_name = dict()
      
      if len(region['objects']) == 0:
        continue

      #region_relaions.append(region['relationships'])
      for obj in range(len(region['objects'])):
        obj_id = region['objects'][obj]['object_id']
        obj_id_to_name[obj_id] = region['objects'][obj]['name']
        if obj_id in obj_to_attr:
          
          region_attributes.append([region['objects'][obj]['name'] ,obj_to_attr[obj_id]])
        region_objects.append(region['objects'][obj]['name'])

      region_phrase.append(region['phrase'])
      for rel in region['relationships']:
        subject_id = rel['subject_id']
        object_id  = rel['object_id']
        predicate  = rel['predicate']
        region_relaions.append([obj_id_to_name[subject_id], predicate, obj_id_to_name[object_id]])
      #print "objects: ", region_objects
      #print "attributes:  ", region_attributes
      #print "phrase:  ", region_phrase
      #print "relations: ", region_relaions, '\n'

      total_region_graphs.append([region_phrase, region_objects, region_attributes, region_relaions])
  
    json.dump(total_region_graphs, open("pre_" + TARGET + "_%d.json"%NUM,"w"), indent=2)

def output_phrases():
  all_region_graphs = read_region_graphs()
  all_phrases = []
  total_images = len(all_region_graphs)
  print("Total images: %d" % total_images)
  # all phrases exclude the sentence that has no objects in scene graph
  fout = codecs.open('all_phrases.txt','w', encoding='utf-8')

  for im in range(total_images):
    
    regions = all_region_graphs[im]['regions']

    total_regions = len(regions)
    count_region = 0
    for region in regions:
      count_region += 1
      print("Progress: images:  %d/%d, regions:  %d/%d" % (im, total_images, count_region, total_regions))

      
      if len(region['objects']) == 0:
        continue

      fout.write(region['phrase']+'\n')


def output():
  #count = 0
  input_text = pickle.load(open("input_text.p", "rb"))
  fout = codecs.open("input.txt", "wb", encoding='utf-8')
  #print input_text
  for i in range(len(input_text)):
    for phrase in range(len(input_text[i])):
      #count += 1
      fout.write(input_text[i][phrase]+"\n")

  #print count

def process_vg():
  all_region_graphs = read_region_graphs()
  all_attributes   = read_attributes()
  image_data = json.load(open('./data/image_data_%d.json' %NUM,'r'))
  assert len(all_region_graphs) == len(image_data)

  vg_coco_id = []
  all_labels = []
  total_images = len(all_region_graphs)

  for im in range(len(all_region_graphs)):

    if image_data[im]['coco_id'] == None:
      continue
    
    regions = all_region_graphs[im]['regions']
    im_attributes = all_attributes[im]['attributes']
    obj_to_attr = dict()
    
    for attr in range(len(im_attributes)):
      obj_id = im_attributes[attr]['object_id']
      try:
        obj_to_attr[obj_id] = im_attributes[attr]['attributes']
      except:
        continue

    total_regions = len(regions)
    count_region = 0
    for region in regions:
      if len(vg_coco_id) != 0:
        if pre_phrase == region['phrase']:
          continue

      count_region += 1
      print("Progress: images:  %d/%d, regions:  %d/%d" % (im, total_images, count_region, total_regions))

      region_objects = []
      region_attributes = []
      region_relaions = []
      region_phrase = []
      obj_id_to_name = dict()
      
      if len(region['objects']) == 0:
        continue

      pre_phrase = region['phrase']
      vg_coco_id.append(image_data[im]['coco_id'])

  pickle.dump(vg_coco_id, open('vg_coco_id_%d.p' % NUM, 'w'))




process_labels()

#process_data()

#output_phrases()

#divide_data()

