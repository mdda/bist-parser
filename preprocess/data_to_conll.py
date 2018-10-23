import numpy as np
import pickle
import json
import utils as ut
import re
import nltk
import codecs
from optparse import OptionParser

NLTK_PATH = "nltk_data"
nltk.data.path.append(NLTK_PATH)
from nltk.corpus import wordnet

parser = OptionParser()
parser.add_option("--input",  dest="input_path",  help="Required Processed input file", metavar="FILE", default="./output/pre_coco_train.json")
parser.add_option("--output", dest="output_path", help="Processed file output file path", metavar="FILE", default="./output/coco_train.conll")
parser.add_option("--train",  dest="isTrain",     help="Check if processed file required Training", default=True)
(options, args) = parser.parse_args()

#all_region_graphs = json.load(codecs.open(datapath, "r", 'utf-8'))
region_graphs_file = codecs.open(options.input_path, "r", 'utf-8')

#data_number = len(all_region_graphs)

fout = codecs.open(options.output_path, 'w', encoding='utf-8')   # already in a sensible line-appending format...
#fout.write('id\tparent_id\trel\tprop\n')

TRAIN = options.isTrain
#SHARE = False

class Node:
  def __init__(self, idx):
    self.id        = idx
    self.parent_id = None
    self.rel       = None
    self.prop      = None
    self.word      = None
    self.synsets   = None

def find(phrasee, word):
  sentence = ' ' + (' '.join(phrasee)) + ' '
  return sentence.find(" "+word+" ")

def find_object(obj, phrasee):
  find_id = find(phrasee, obj)
  if not (find_id + 1):
    return False, []
  else:
    sentence = ' ' + ' '.join(phrasee) + ' '
    temp = sentence.replace(" "+obj+" ", " TEMPTAG ")
    temp = temp.split()
    head_id = temp.index("TEMPTAG")
    return True, [word_id for word_id in range( head_id, head_id+len(obj.split()) )]


def find_wn(node_list, word):
  if len(word.split()) > 1:
    word = '_'.join(word.split())

  word_wn = ut.word_to_wn(word)
  max_id = 0
  max_lap = 0

  for word_id in range(len(node_list)):
    if node_list[word_id].prop != None:
      continue
    #train_word_wn = ut.word_to_wn(sentence[word_id])
    overlap = ut.similar_to(word_wn, node_list[word_id].synsets)
    if len(overlap) > max_lap:
      max_lap = len(overlap)
      max_id = word_id
      #print "wn synsets", node_list[word_id].synsets
      #print "wn id: ", word_id
      #print "wn input word:  ", word
      #print "wn input word syn", word_wn
      return (True, word_id)
      
  if max_lap > 0:
    return True, max_id
  else:
    return False, None


def find_pos(phrase_sen, word):
  #print 'first phrase sentence: ', phrase_sen 
  phrase_sen = ' ' + phrase_sen + ' '
  
  #print phrase_sen
  #print phrase_sen.find(word)
  
  temp = phrase_sen.replace(" "+word+" ", " TEMPTAG ")
  temp = temp.split()
  
  #print "word: ", word
  #print "temp: ", temp
  #print "len temp: ", len(temp)
  
  return temp.index('TEMPTAG') + len(word.split())-1


def find_pos_wn(node_list, word, prop): #especially for finding objects
  if len(word.split()) > 1:
    word = '_'.join(word.split())
  word_syn = ut.word_to_wn(word)

  max_lap = 0
  max_id = 0

  for node_id in range(len(node_list)):
    #print "input word;   ",word
    #print node_list[node_id].word
    #print node_list[node_id].synsets
    #print word_syn
    #if node_list[node_id].prop != prop: 
    # print "input worddd:   ", word
    # print node_list[node_id].prop
    # print "node word:  ", node_list[node_id].word
    # continue
    
    overlap = ut.similar_to(node_list[node_id].synsets, word_syn)
    if len(overlap) > max_lap:
      max_lap = len(overlap)
      max_id  = node_id
      #print "input wordddd: ", word
      #print node_id
      
  if max_lap>0:
    return max_id
  else:
    return None


def lower_tuples(tuples, prop):   # OMG ugly...
  if   prop == 0:
    return [word.lower() for word in tuples]

  elif prop == 1:
    return [word.lower() for word in tuples]

  elif prop == 2:
    return[[attr_pair[0].lower(),[attr.lower() for attr in attr_pair[1]]] for attr_pair in tuples  ]

  else:
    return[[rels_pair[0].lower(), rels_pair[1].lower(), rels_pair[2].lower()] for rels_pair in tuples]


#print "Number of Data:  ", data_number

#for data_id in range(data_number):
for data_id, region_graphs_json in enumerate(region_graphs_file):
  print("No. %d sentence" % (data_id,) )
  region_graphs_data_id = json.loads(region_graphs_json)
  
  conll = dict()
  vocab = []
  vocab_to_id = dict()
  node_list = []
  obj_set = set()

  # Clean up the phrase 
  input_sent = re.sub( '"',' ', ' '.join( region_graphs_data_id[0][0].lower().split() ))
  input_sent = ' '.join( input_sent.split() )
  phrase_sen = " " + input_sent + " "
  phrase     = lower_tuples(input_sent.split(), 0)
  objects    = lower_tuples(region_graphs_data_id[1], 1)
  attributes = lower_tuples(region_graphs_data_id[2], 2)
  relations  = lower_tuples(region_graphs_data_id[3], 3)

  #phrase  = phrase.split()
  phrasee = phrase[:]


  # Build a basic empty word structure (cross-reference duplicate words)
  for word_id in range(len(phrase)):
    node         = Node(word_id+1)
    node.word    = phrase[word_id]
    node.synsets = ut.word_to_wn(node.word)

    node_list.append(node)
    if phrase[word_id] in vocab_to_id:
      vocab_to_id[phrase[word_id]].append(word_id)
    else:
      vocab_to_id[phrase[word_id]] = [word_id]

  # First round of Algorithm : Objects
  for obj in objects:
    #obj = ' '.join(obj.split())
    #print "obj: ",obj
    #print phrasee
    (isExist, id_list) = find_object(obj, phrasee)
    if isExist:
      #print "find obj: ", obj
      obj_set.add(' '.join(obj.split()))
      for word_idx in range(len(id_list)):
        if word_idx != len(id_list)-1:
          node_list[id_list[word_idx]].parent_id = node_list[id_list[word_idx+1]].id
          node_list[id_list[word_idx]].rel       = 'same'

        node_list[id_list[word_idx]].prop = "OBJ"
        phrasee[id_list[word_idx]]        = "OBJ_" + str(word_idx)
  #print "OBJ list:  ", objects

  # First round of Algorithm : Attributes
  for attr_pair in attributes:
    if attr_pair[0] in obj_set:
      found_idx = find_pos(phrase_sen, attr_pair[0])
    else:
      continue
      
    for attr in attr_pair[1]:
      #print "join sent:  ", ' '.join(phrasee)
      if (find(phrasee, attr)+1):
        #print "find attr:  ", attr
        #print "find obj: ", attr_pair[0]
        #temp_attr = attr
        #attr_pair.remove(attr)
        attr_tail_id = find_pos(' '.join(phrasee), attr)

        node_list[attr_tail_id].parent_id = node_list[found_idx].id
        node_list[attr_tail_id].rel  = "ATTR"
        node_list[attr_tail_id].prop = "ATTR"
        for attr_id in xrange(len(attr.split())-1, 0, -1):
          node_list[attr_tail_id - attr_id].parent_id = node_list[attr_tail_id - attr_id +1].id
          node_list[attr_tail_id - attr_id].rel       = 'same'
          phrasee[attr_tail_id - attr_id] = 'ATTR_%d' % (len(attr.split()) - attr_id - 1)

        phrasee[attr_tail_id] = 'ATTR_%d' % (len(attr.split()) - 1)

      else:
        (isATTR, idx) = find_wn(node_list, attr)
        if isATTR:
          #print "find attr:  ", attr
          #print "find obj: ", attr_pair[0]
          node_list[idx].parent_id = node_list[found_idx].id
          node_list[idx].rel  = "ATTR"
          node_list[idx].prop = "ATTR"
          phrasee[idx] = "ATTR_0"


  # First round of Algorithm : Relations
  for rel_pair in relations:
    sub  = ' '.join(rel_pair[0].split())
    obj  = ' '.join(rel_pair[2].split())
    pred = rel_pair[1]
    if (sub not in obj_set) or (obj not in obj_set):
      continue

    #print "node list prop: ", node_list[1].prop
    
    sub_idx = find_pos(phrase_sen, sub)
    obj_idx = find_pos(phrase_sen, obj)
    
    if (find(phrasee, pred)+1):
      #print "looking pred: ", pred
      pred_tail_id = find_pos(phrase_sen, pred)
      if node_list[pred_tail_id].prop != None and node_list[pred_tail_id].prop != "PRED":
        continue
      for pred_id in xrange(len(pred.split())-1,0,-1):
        node_list[pred_tail_id - pred_id].parent_id = node_list[pred_tail_id - pred_id +1].id
        node_list[pred_tail_id - pred_id].rel       = 'same'
        
    else:
      (isPred ,pred_tail_id) = find_wn(node_list, rel_pair[1])
      if not isPred:
        continue
    
    #print "pred id:  ", pred_tail_id
    #print "sentence:  ", phrasee
    node_list[pred_tail_id].prop = "PRED"
    phrasee[pred_tail_id] = "PRED"
    #print "PPRREEDD word:  ", node_list[pred_tail_id].word
    #print "REL PAIR 1:", rel_pair[1]
    
    #node_list[sub_idx].rel = "SUB"
    node_list[pred_tail_id].rel = "PRED"
    node_list[obj_idx].rel = "OBJT"

    #node_list[sub_idx].parent_id = node_list[pred_tail_id].id
    node_list[pred_tail_id].parent_id = node_list[sub_idx].id
    node_list[obj_idx].parent_id = node_list[pred_tail_id].id


  # Second round of Algorithm : Objects
  #print "phrasee before second object:  ", phrasee
  for obj in objects:
    if obj in obj_set:
      continue
      
    else:
      (isObj, idx) = find_wn(node_list, obj)
      if isObj:
        if node_list[idx].prop != None:
          continue
        phrasee[idx] = "OBJ_000"
        node_list[idx].prop = "OBJ"
        
        #print "obj list word: ", node_list[idx].word
        #print "obj list id:  ", node_list[idx].id
        obj_set.add(obj)


  # Second round of Algorithm : Attributes
  for attr_pair in attributes:
    if (find(phrase_sen.split(), attr_pair[0])+1):
      found_idx = find_pos(phrase_sen, attr_pair[0])

    else:
      found_idx = find_pos_wn(node_list, attr_pair[0], "OBJ")
      if not isinstance(found_idx, int):
        continue

    for attr in attr_pair[1]:
      #print "join sent:  ", ' '.join(phrasee)
      if (find(phrasee, attr)+1):
        #print "find attr:  ", attr
        #print "find obj: ", attr_pair[0]
        #temp_attr = attr
        #attr_pair.remove(attr)
        attr_tail_id = find_pos(' '.join(phrasee), attr)
        #print "tail id: ", attr_tail_id

        node_list[attr_tail_id].parent_id = node_list[found_idx].id
        node_list[attr_tail_id].rel  = "ATTR"
        node_list[attr_tail_id].prop = "ATTR"
        for attr_id in xrange(len(attr.split())-1, 0, -1):
          node_list[attr_tail_id - attr_id].parent_id = node_list[attr_tail_id - attr_id +1].id
          node_list[attr_tail_id - attr_id].rel       = 'same'
          phrasee[attr_tail_id - attr_id] = 'ATTR_%d' % (len(attr.split()) - attr_id - 1)

        phrasee[attr_tail_id] = 'ATTR_%d' % (len(attr.split()) - 1)

      else:
        (isATTR, idx) = find_wn(node_list, attr)
        if isATTR:
          #print "find attr:  ", attr
          #print "find obj: ", attr_pair[0]
          node_list[idx].parent_id = node_list[found_idx].id
          node_list[idx].rel  = "ATTR"
          node_list[idx].prop = "ATTR"
          phrasee[idx] = "ATTR_0"


  # Second round of Algorithm : Relations
  for rel_pair in relations:
    sub  = ' '.join(rel_pair[0].split())
    obj  = ' '.join(rel_pair[2].split())
    pred = rel_pair[1]
    if (sub not in obj_set) or (obj not in obj_set):
      continue

    #print "phrase_sen  in rel: ", phrase_sen
    #print "sub in rel: ", sub
    if (find(phrase_sen.split(), sub)+1):
      sub_idx = find_pos(phrase_sen, sub)
    else:
      sub_idx = find_pos_wn(node_list, sub, "OBJ")
      
      if not isinstance(sub_idx, int):
        print("error subjs= word: ", sub)
        print(sub)
        print(obj_set)
        print(phrase_sen)
        print(phrasee)
        print(node_list[1].prop)
        print(node_list[0].prop)
        print(attributes)
        print(objects)
        print(relations)
        print(ut.similar(node_list[2].synsets, ut.word_to_wn(sub)))
        exit()
        
    if (find(phrase_sen.split(), obj)+1):
      obj_idx = find_pos(phrase_sen, obj)
    else:
      obj_idx = find_pos_wn(node_list, obj, "OBJ")
      
      if not isinstance(obj_idx, int):
        print("error objs= word: ", obj)
        print(obj)
        print(obj_set)
        print(phrase_sen)
        print(phrasee)
        print(node_list[1].prop)
        print(relations)
        print(objects)
        print(ut.similar(node_list[1].synsets, ut.word_to_wn(sub)))
        exit()
    
    if (find(phrasee, pred)+1):
      pred_tail_id = find_pos(phrase_sen, pred)
      if node_list[pred_tail_id].prop != None and node_list[pred_tail_id].prop != "PRED":
        continue
        
      for pred_id in xrange(len(pred.split())-1,0,-1):
        node_list[pred_tail_id - pred_id].parent_id = node_list[pred_tail_id - pred_id +1].id
        node_list[pred_tail_id - pred_id].rel       = 'same'
        
    else:
      (isPred ,pred_tail_id) = find_wn(node_list, rel_pair[1])
      if not isPred:
        continue
    
    #print "pred id:  ", pred_tail_id
    #print "sentence:  ", phrasee
    
    node_list[pred_tail_id].prop = "PRED"
    phrasee[pred_tail_id] = "PRED"
    
    #print "PPRREEDD word:  ", node_list[pred_tail_id].word
    #print "REL PAIR 1:", rel_pair[1]
    
    #node_list[sub_idx].rel = "SUB"
    node_list[pred_tail_id].rel = "PRED"
    node_list[obj_idx].rel = "OBJT"

    #node_list[sub_idx].parent_id = node_list[pred_tail_id].id
    node_list[pred_tail_id].parent_id = node_list[sub_idx].id
    node_list[obj_idx].parent_id = node_list[pred_tail_id].id

  isDuplicate = False
  for node in node_list:
    if node.prop == 'OBJ' and node.parent_id == None:
      node.parent_id = 0
      #node.rel = 'OBJ'
      
    if TRAIN:   #turn off when generate dev and test conll
      if node.parent_id == node.id: 
        isDuplicate = True
        break

  if isDuplicate:
    continue  # Skip this one - only do this for the training set...

  for node in node_list:
    fout.write(str(node.id))
    fout.write("\t"+node.word)
    fout.write("\t"+(str(node.parent_id) if node.parent_id != None else '_')) 
    fout.write("\t"+(str(node.rel) if node.rel != None else '_'))
    fout.write("\t"+(str(node.prop) if node.prop != None else '_')+'\n')
  fout.write('\n')
  
  #exit()
  
print("Finished Alignment")

