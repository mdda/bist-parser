from nltk.corpus import wordnet

def word_to_wn(word):
  synsets = []
  for syns in wordnet.synsets(word):
    for l in syns.lemmas():
      synsets.append(l.name())

  return set(synsets)

def similar(pred_syns, ref_syns):
  if len( pred_syns.intersection(ref_syns) )>0:
    return True
  else:
    return False

def similar_to(pred_syns, ref_syns):
  return pred_syns.intersection(ref_syns)
