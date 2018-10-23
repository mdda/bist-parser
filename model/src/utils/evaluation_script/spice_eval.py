import io
import sys

import json
import codecs

import spice_wordnet as sw

import numpy as np

import argparse

import nltk
nltk.data.path.append('nltk_data')
from nltk.corpus import wordnet

class Node:
  def __init__(self, id, word, parent_id, relation):
    self.id        = id
    self.word      = word
    self.parent_id = parent_id
    self.relation  = relation

class SemanticTuple(object):
  def __init__(self, word):
    self.word = ' '.join(word.strip().lower().split())
    self.word_to_synset()

  def word_to_synset(self):
    lemma_synset = []
    word_split = self.word.split()
    
    if len(word_split) >= 2:
      self.word = "_".join(word_split)
      
    lemma_synset.append(self.word)

    for sys in wordnet.synsets(self.word):
      for l in sys.lemmas():
        lemma_synset.append(l.name())

    self.lemma_synset = set(lemma_synset)


def similar(tup_syns, pred):
  if len(tup_syns) != len(pred): 
    return False

  else:
    for w_id in range(len(tup_syns)):
      #print("w_id:  ", w_id)
      if len(tup_syns[w_id].intersection(pred[w_id])) == 0:
        return False
    return True


def find_tuples(node_list):
  tuples = []
  objects         = []
  objects_tail_id = []
  OBJTs_id = [] 

  attrs         = []
  attrs_tail_id = []

  preds         = []
  preds_tail_id = []

  
  for id_rnode, rnode in zip(range(len(node_list)-1,-1,-1), reversed(node_list)):
    if rnode.parent_id == -1:
      continue
      
    if rnode.relation == 'OBJT' or rnode.parent_id == 0:
      objects_tail_id.append(rnode.id)

      if rnode.relation == 'OBJT':
        if node_list[rnode.parent_id-1].relation == 'PRED':
          OBJTs_id.append(rnode.id)
      if rnode.id != 1 and node_list[id_rnode-1].relation == 'same':
        obj = [rnode.word]

        rnode_next = node_list[id_rnode-1]
        while True:
          obj.insert(0, rnode_next.word)
          if rnode_next.id == 1:
            break
          rnode_next = node_list[rnode_next.id-1-1]
          if rnode_next.relation != 'same':
            break
            
        objects.append(obj)
        
      else:
        objects.append([rnode.word])

  for id_rnode, rnode in zip(range(len(node_list)-1, -1, -1), reversed(node_list)):
    if rnode.relation == 'ATTR' or rnode.relation == 'PRED':
      if rnode.relation == 'ATTR':
        attrs_tail_id.append(rnode.id)
        if rnode.id != 1 and node_list[id_rnode-1].relation == 'same':
          attr = [rnode.word]
          rnode_next = node_list[id_rnode-1]
          while True:
            attr.insert(0, rnode_next.word)
            if rnode_next.id == 1:
              break
              
            rnode_next = node_list[rnode_next.id-1-1]
            if rnode_next.relation != 'same':
              break
              
          attrs.append(attr)
          
        else:
          attrs.append([rnode.word])

      else:
        preds_tail_id.append(rnode.id)
        if rnode.id != 1 and node_list[id_rnode-1].relation == 'same':
          pred = [rnode.word]
          rnode_next = node_list[id_rnode-1]
          while True:
            pred.insert(0, rnode_next.word)
            if rnode_next.id == 1:
              break
              
            rnode_next = node_list[rnode_next.id-1-1]
            if rnode_next.relation != 'same':
              break
              
          preds.append(pred)
          
        else:
          preds.append([rnode.word])

  for obj in objects:
    tuples.append([' '.join(obj)])

  print(preds_tail_id)

  for attr_id, attr in enumerate(attrs):
    comp_attr = ' '.join(attr)
    obj_id = node_list[attrs_tail_id[attr_id]-1].parent_id
    
    if obj_id not in objects_tail_id:
      print("attr object error")
      comp_obj = node_list[obj_id-1].word
      
    else:
      obj = objects_tail_id.index(obj_id)
      comp_obj = ' '.join(objects[obj])
      
    tuples.append((comp_obj, comp_attr))

  for OBJT_id in OBJTs_id:
    obj = objects_tail_id.index(OBJT_id)
    comp_obj = ' '.join(objects[obj])

    pred_id = node_list[OBJT_id-1].parent_id
    pred = preds_tail_id.index(pred_id)
    comp_pred = ' '.join(preds[pred])

    sub_id = node_list[pred_id-1].parent_id
    try:
      sub    = objects_tail_id.index(sub_id)
      comp_sub = ' '.join(objects[sub])
    except:
      comp_sub = node_list[sub_id-1].word
    
    tuples.append((comp_sub, comp_pred, comp_obj))

  return tuples

  
def get_tuples(sent):
  node_list = []
  for word in sent:
    word = word.split('\t')
    #if word[2] == -1:
    # continue
    
    try:
      node = Node(int(word[0]), word[1], int(word[2]), word[3])
    except:
      print sent
      print word
      node = Node(int(word[0]), word[1], -1, word[3])
      
    node_list.append(node)

  tuples = find_tuples(node_list)
  return tuples


def evaluate_ospice(spice_tuple, ref_tuple):
  count_tuple = 0

  num_ref   = len(ref_tuple)
  num_pred = len(spice_tuple)

  spice_wordnet = []

  for tup in spice_tuple:
    tup_syns = []
    for word in tup:
      st = SemanticTuple(word)
      tup_syns.append(st.lemma_synset)

    spice_wordnet.append(tuple(tup_syns))

  for tup in ref_tuple:
    tup_syns = []

    for word in tup:
      st = SemanticTuple(word)
      tup_syns.append(st.lemma_synset)

    for pred in spice_wordnet:
      if similar(tup_syns, pred):
        #print "similar: ", tup_syns, pred
        count_tuple += 1
        break

  if num_pred == 0:
    p_score = 0
  else:
    p_score = count_tuple/float(num_pred)

  s_score = count_tuple/float(num_ref)

  if count_tuple == 0:
    sg_score = 0
  else:
    sg_score = 2*p_score*s_score/(p_score+s_score)

  if False and sg_score > 1:
    #print(ref_tuple)
    #print(spice_wordnet)
    print(len(ref_tuple))
    print(len(spice_wordnet))
    print(p_score)
    print(s_score)
    print(sg_score)
    print("NOPE")
    exit()

  return sg_score


def evaluate_spice(spice_tuple, ref_tuple):
  count_tuple = 0

  spice_predict_tuple = spice_tuple[:]
  
  num_ref   = len(ref_tuple)
  num_pred  = len(spice_tuple)
  check_ref  = np.zeros( (num_ref,) )
  check_pred = np.zeros( (num_pred,) )

  ans = []
  for tup_id, tup in enumerate(ref_tuple):
    for spice_id, spice_tup in enumerate(spice_tuple):
      if check_pred[spice_id]==0 and tup==spice_tup:
        ans.append(tup)
        check_ref[tup_id] = 1
        check_pred[spice_id] = 1
        count_tuple += 1
        break   

  spice_wordnet = []
  for tup_id, tup in enumerate(spice_tuple):
    tup_syns = []
    if check_pred[tup_id] != 1:
      for word in tup:
        st = SemanticTuple(word)
        tup_syns.append(st.lemma_synset)

    spice_wordnet.append(tuple(tup_syns))
  

  for tup_id, tup in enumerate(ref_tuple):
    if check_ref[tup_id] == 1:
      continue
    tup_syns = []

    for word in tup:
      st = SemanticTuple(word)
      tup_syns.append(st.lemma_synset)

    for pred_id, pred in enumerate(spice_wordnet):
      if check_pred[pred_id]==0 and similar(tup_syns, pred):
        count_tuple += 1 
        check_ref[tup_id]   = 1
        check_pred[pred_id] = 1
        break
      

  if num_pred == 0:
    p_score = 0
  else:
    p_score = count_tuple/float(num_pred)

  s_score = count_tuple/float(num_ref)

  if count_tuple == 0:
    sg_score = 0
  else:
    sg_score = 2*p_score*s_score/(p_score+s_score)

  if True and sg_score > 1:
    #print(ref_tuple)
    #print(spice_wordnet)
    print(len(ref_tuple))
    print(len(spice_wordnet))
    print(p_score)
    print(s_score)
    print(sg_score)
    print("NOPE")
    exit()

  return sg_score


def check_len(conll_path, gold_path):
  count_gold = 0
  refs = json.load(open(gold_path, 'r'))
  f    = codecs.open(conll_path, 'r', encoding='utf-8')

  for line in f.readlines():
    line = line.strip()
    if line == '':
      count_gold += 1
      
  print(count_gold)
  print(len(refs))

  assert count_gold == len(refs)


def read_conll(conll_path, gold_path):
  print(gold_path)
  print(conll_path)
  
  fout = open('spice.txt', 'w')
  refs = json.load(open(gold_path, 'r'))
  f    = codecs.open(conll_path, 'r', encoding='utf-8')
  
  #index = [3061, 4683, 6026, 6326, 6592, 6855, 7031, 7042, 7638, 8151, 8153, 8207, 8384, 9780, 12153]
  #index = [2573, 14812, 33807, 82665, 150294, 195522]
  #index  = [8550, 12570, 12863, 14805, 15976, 20164]
  #with open('index.txt', 'r') as fin:
  # for line in fin.readlines

  print(len(refs))
  #print len(f)
  #assert len(refs) == len(f)

  s_score = 0
  sent = []
  count_gold = 0

  with open('dep_parse_08_04.txt', 'w') as fdep:
    for line in f.readlines():
      line = line.strip()
      if line == '':
        predict_tuples = get_tuples(sent)
        ref_tuples     = sw.label_data(refs[count_gold])
        print("PREDICT tuples:\t", predict_tuples)
        print("REFERENCE tuples:\t", ref_tuples)
        if count_gold in index:
          fdep.write("id: "+str(count_gold)+ ' ')
          fdep.write("PRED: "+str(predict_tuples))
          fdep.write(' ')
          fdep.write("REF: "+str(ref_tuples))
          fdep.write('\n')

        spice_score    = evaluate_spice(predict_tuples, ref_tuples)
        print(spice_score)
        
        fout.write(str(spice_score)+'\n')
        s_score += spice_score
        count_gold += 1
        sent = []

      else:
        sent.append(line)
        
  print(count_gold)
  print(len(refs))

  assert count_gold == len(refs)

  print("Num of prediction: ", count_gold)

  return s_score/float(count_gold)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("gold_file", type=str,
            help="Name of the CoNLL-U file with the gold data.")
  parser.add_argument("prediction_file", type=str,
            help="Name of the CoNLL-U file with the predicted data.")
            
  #parser.add_argument("epoch", type=str,
  #                    help="Name of the CoNLL-U file with the predicted data.")

  args = parser.parse_args()

  check_len(args.prediction_file, args.gold_file)

  s_score = read_conll(args.prediction_file, args.gold_file)
  print("SPICE score:\t %.4f" % (s_score))

if __name__ == '__main__':
  main()

