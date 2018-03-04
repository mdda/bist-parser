import json
import spice_wordnet as sw
import argparse
import nltk
import os
import numpy as np
nltk.data.path.append('nltk_data')
from nltk.corpus import wordnet

class RG(object):

	def __init__(self, objects, attributes, relationships):
		self.objects = objects
		self.attributes = attributes
		self.relationships = relationships
		self.graph_tuple = []
		self.convert_tuple()


	def convert_tuple(self):
		# objects -> tuple
		for i in range(len(self.objects)):
			temp_obj = self.objects[i]["names"][0]
			self.graph_tuple.append([temp_obj])

		#attributes -> tuple
		for i in range(len(self.attributes)):
			sub_id = self.attributes[i]['subject']
			subject = self.objects[sub_id]['names'][0]
			attribute = self.attributes[i]['attribute']
			self.graph_tuple.append(tuple([subject, attribute]))

		for i in range(len(self.relationships)):
			sub_id = self.relationships[i]['subject']
			obj_id = self.relationships[i]['object']
			sub = self.objects[sub_id]["names"][0]
			obj = self.objects[obj_id]["names"][0]
			self.graph_tuple.append(tuple([sub, self.relationships[i]['predicate'], obj]))

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


def read_pred(num, sys_path):
	stanford_tuples = []
	for i in range(num):
		file_name = "output_json_"+str(i)+".json"
		predict_path = os.path.join(sys_path, file_name)
		predict = json.load(open(predict_path, "r"))
		stanford_tuples.append(predict)

	return stanford_tuples

def stanford_format_eval(fstan, pred, ref):

	count_tuple = 0

	ref_tuple = sw.label_data(ref)
	spice_tuple = RG(pred['objects'], pred['attributes'], pred['relationships']).graph_tuple

	fstan.write("Stanford Pred tuple: "+str(spice_tuple))
	fstan.write("\nREF tupels:  "+str(ref_tuple))
	print "Stanford Pred tuple: ", spice_tuple
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

	print 'spice score: ', sg_score
	fstan.write('spice score: '+str(sg_score)+'\n\n')


	return sg_score












def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("gold_file", type=str,
						help="Name of the CoNLL-U file with the gold data.")
	parser.add_argument("sys_path", type=str,
						help="Name of the CoNLL-U file with the predicted data.")
    #parser.add_argument("epoch", type=str,
    #                    help="Name of the CoNLL-U file with the predicted data.")

	args = parser.parse_args()
	refs  = json.load(open(args.gold_file, 'r'))
	preds = read_pred(len(refs), args.sys_path)

	s_score = 0
	fstan = open('stanford_tuples.txt', 'w')

	with open('stanford_score.txt', 'w') as fspice: 

		for stanford_tuple, ref_tuple in zip(preds, refs): 
			sg_score = stanford_format_eval(fstan, stanford_tuple, ref_tuple)
			s_score += sg_score
			fspice.write(str(sg_score)+'\n')
			


		print "Stanford Parser Score:\t%.4f" %(s_score/float(len(preds)))

	#check_len(args.system_file, args.gold_file)

	#s_score = read_conll(args.system_file, args.gold_file)
	#print "Stanford score:\t %.4f" % (s_score)

if __name__ == '__main__':
	main()



