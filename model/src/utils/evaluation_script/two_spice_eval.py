import json
import spice_wordnet as sw
import argparse
import nltk
import numpy as np
nltk.data.path.append('/media/Work_HD/yswang/nltk_data')
from nltk.corpus import wordnet

class SemanticTuple(object):

	def __init__(self, word):

		self.word = word
		self.word_to_synset()


	def word_to_synset(self):

		lemma_synset = []
		word_split = self.word.split()
		if len(word_split) >= 2:
			self.word = "_".join(word_split)

		for sys in wordnet.synsets(self.word):
			for l in sys.lemmas():
				lemma_synset.append(l.name())

		self.lemma_synset = set(lemma_synset)


def similar(tup_syns, pred):
	if len(tup_syns) != len(pred): 
		return False
	else:
		for w_id in range(len(tup_syns)):
			#print "w_id:  ", w_id
			
			if len(tup_syns[w_id].intersection(pred[w_id])) == 0:
				return False
		return True

def ospice_format_eval(spice_file, ref):

	count_tuple = 0
	ref_tuple = sw.label_data(ref)
	
	#### spice tuple #####
	spice_tuple = []
	for tup in spice_file['test_tuples']:
		if len(tup['tuple']) == 1:
			spice_tuple.append(tup['tuple'])
		else:
			spice_tuple.append(tuple(tup['tuple']))
	######################

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

	return sg_score


def spice_format_eval(spice_file, ref):
	
	count_tuple = 0
	ref_tuple = sw.label_data(ref)

	#### spice tuple #####
	spice_tuple = []
	for tup in spice_file['test_tuples']:
		if len(tup['tuple']) == 1:
			spice_tuple.append(tup['tuple'])
		else:
			spice_tuple.append(tuple(tup['tuple']))
	######################

	print "SPICE Pred tuple: ", spice_tuple
	print "REF tupels:  ", ref_tuple

	num_ref   = len(ref_tuple)
	num_pred = len(spice_tuple)
	check_pred = np.zeros((num_pred))

	ans = []
	for tup in ref_tuple:
		for spice_id, spice_tup in enumerate(spice_tuple):
			if check_pred[spice_id]==0 and tup==spice_tup:
				ans.append(tup)
				check_pred[spice_id] = 1
				count_tuple += 1
				break
	[ref_tuple.remove(x) for x in ans]
		

	spice_wordnet = []

	for tup_id, tup in enumerate(spice_tuple):
		tup_syns = []
		if check_pred[tup_id] != 1:
			for word in tup:
				st = SemanticTuple(word)
				tup_syns.append(st.lemma_synset)

		spice_wordnet.append(tuple(tup_syns))

	for tup in ref_tuple:
		tup_syns = []

		for word in tup:
			st = SemanticTuple(word)
			tup_syns.append(st.lemma_synset)

		for pred_id, pred in enumerate(spice_wordnet):
			if check_pred[pred_id]==0 and similar(tup_syns, pred):
				count_tuple += 1
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

	return sg_score

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("gold_file", type=str,
						help="Name of the CoNLL-U file with the gold data.")
	parser.add_argument("system_file", type=str,
						help="Name of the CoNLL-U file with the predicted data.")

	args = parser.parse_args()

	preds = json.load(open(args.system_file,'r'))
	refs  = json.load(open(args.gold_file, 'r'))

	s_score = 0
	print len(refs)
	print len(preds)
	assert len(refs) == len(preds)

	with open('spice_score.txt', 'w') as fspice: 

		for spice_tuple, ref_tuple in zip(preds, refs): 
			sg_score = spice_format_eval(spice_tuple, ref_tuple)
			s_score += sg_score
			fspice.write(str(sg_score)+'\n')
			#fspice.write()


		print "SPICE Parser Score:\t%.4f" %(s_score/float(len(preds)))

if __name__ == '__main__':
	main()