import pickle
import cPickle as cp
import json
import codecs
import sys

#NUM    = int(sys.argv[1])  # range: 0~9
TARGET = sys.argv[2]       # coco_train or coco_dev


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
  all_attributes    = read_attributes()
  all_region_graphs = read_region_graphs()

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
      region_relations = []
      region_phrase = []
      obj_id_to_name = dict()
      
      if len(region['objects']) == 0:
        continue

      #region_relations.append(region['relationships'])
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
        region_relations.append([obj_id_to_name[subject_id], predicate, obj_id_to_name[object_id]])
        
      #print "objects: ", region_objects
      #print "attributes:  ", region_attributes
      #print "phrase:  ", region_phrase
      #print "relations: ", region_relations, '\n'

      total_region_graphs.append([region_phrase, region_objects, region_attributes, region_relations])
    
    # This seems to be indented too far...
    json.dump(total_region_graphs, open("pre_" + TARGET + "_%d.json"%NUM,"w"), indent=2)


def process_labels_rowwise():
  #all_attributes    = read_attributes()
  #all_region_graphs = read_region_graphs()

  #all_labels = []  # unused
  
  # Find the length of the image_data file...
  #with open('./data/image_data.json.rows', 'r') as f:
  #  total_image_count = len( f.readlines() )
  ###total_images = len(all_region_graphs)
  #print("Total images: %d" % total_image_count)
  
  with open('./intermediate/'+TARGET+"_attr.json.rows", 'r') as f:
    total_image_count = len( f.readlines() )
  print("Total images: %d" % total_image_count)

  attributes_file    = open('./intermediate/'+TARGET+"_attr.json.rows", 'r')    # Not using NUM now...
  region_graphs_file = open('./intermediate/'+TARGET+"_region.json.rows", 'r')  # Not using NUM now...
  
  # Cross-fingers
  #assert len(all_region_graphs) == len(all_attributes)
  
  regions_output = open("./output/pre_" + TARGET + ".json.rows", "w")
  
  #total_region_graphs = []
  #for im in range(len(all_region_graphs)):
  for im, (attributes_json, region_graph_json) in enumerate(zip(attributes_file, region_graphs_file)):
    # Load 1 row at a time for processing

    region_graphs_im = json.loads(region_graph_json)
    regions = region_graphs_im['regions']

    attributes_im   = json.loads(attributes_json)
    im_attributes = attributes_im['attributes']
    
    # Bring in the attributes object (reorganised) - this tracks all object_id -> attributes in image (across regions)
    obj_to_attr = dict()
    for attr in range(len(im_attributes)):
      obj_id = im_attributes[attr]['object_id']
      try:
        obj_to_attr[obj_id] = im_attributes[attr]['attributes']
      except:
        continue

    region_graphs = []  # Do this for each image now

    total_regions = len(regions)
    count_region = 0
    for region in regions:
      if len(region_graphs)>0:
        if len(region['phrase'].strip().split()) == 0:         # Skip if there's no text
          continue
        if region_graphs[-1][0][0] == region['phrase']:  # Skip if the phrase for this region is the same as the last (for this image)
          continue

      count_region += 1
      print("Progress: images:  %d/%d, regions:  %d/%d" % (im, total_image_count, count_region, total_regions))

      obj_id_to_name = dict() # this tracks all object_id -> names in single region
      
      if len(region['objects']) == 0:
        continue   # Skip if there are no objects in the region
      
      region_objects, region_attributes = [], []
      for obj in range(len(region['objects'])):
        obj_id = region['objects'][obj]['object_id']
        obj_id_to_name[obj_id] = region['objects'][obj]['name']
        if obj_id in obj_to_attr:
          region_attributes.append( [region['objects'][obj]['name'] ,obj_to_attr[obj_id]] )
        region_objects.append( region['objects'][obj]['name'] )

      region_phrases = []
      region_phrases.append(region['phrase'])
      
      region_relations = []
      #region_relations.append(region['relationships'])
      for rel in region['relationships']:
        subject_id = rel['subject_id']
        object_id  = rel['object_id']
        predicate  = rel['predicate']
        region_relations.append( [obj_id_to_name[subject_id], predicate, obj_id_to_name[object_id]] )
        
      #print "objects: ", region_objects
      #print "attributes:  ", region_attributes
      #print "phrase:  ", region_phrase
      #print "relations: ", region_relations, '\n'

      #total_region_graphs.append( [region_phrases, region_objects, region_attributes, region_relations] )
      region_graphs.append( [region_phrases, region_objects, region_attributes, region_relations] )
  
    # Now dump out the region_graphs for just this this image
    for region_data in region_graphs:
      regions_output.write( json.dumps( region_data, separators=(',', ':')) )  # This is compact - and in 1 row
      regions_output.write( "\n" )


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



#process_labels()
process_labels_rowwise()

#process_data()
#output_phrases()
#divide_data()

