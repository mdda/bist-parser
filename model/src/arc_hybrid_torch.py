import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import *
from torch import optim
from torch.autograd import Variable
from utils import ParseForest, read_conll, write_conll
from operator import itemgetter
from itertools import chain
import utils, time, random
import numpy as np

import os
import gc

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

'''exit()
if 'GPU' not in os.environ or int(os.environ['GPU']) == 0:
    print 'Using CPU'
    use_gpu = False
else:
    print 'Using GPU'
    use_gpu = True'''
use_gpu = True

get_data = (lambda x: x.data.cpu()) if use_gpu else (lambda x: x.data)

def Parameter(shape=None, init=xavier_uniform): 
    if hasattr(init, 'shape'):
        assert not shape
        return nn.Parameter(torch.Tensor(init))

    shape = (1, shape) if type(shape) == int else shape
   
    return nn.Parameter(init(torch.Tensor(*shape)))


def scalar(f):
    if type(f) == int:
        if use_gpu:
            return Variable(torch.cuda.LongTensor([f]))
        else:
            return Variable(torch.LongTensor([f]))
    if type(f) == float:
        if use_gpu:
            return Variable(torch.cuda.FloatTensor([f]))
        else:
            return Variable(torch.FloatTensor([f]))

def cat(l, dimension=-1):
   
    valid_l = filter(lambda x: x is not None, l)
   

    if dimension < 0:
        dimension += len(valid_l[0].size())
    try:
        return torch.cat(valid_l, dimension)
    except:
        for ele in l:
            print ele.size()
        print "\n",
        for ele in valid_l:
            print ele.size()
        exit()


class RNNState():
    def __init__(self, cell, hidden=None):
        self.cell = cell
        self.hidden = hidden
        if not hidden:
            if use_gpu:
                self.hidden = Variable(torch.zeros(1, self.cell.hidden_size)).cuda(), \
                          Variable(torch.zeros(1, self.cell.hidden_size)).cuda()
            else:
                self.hidden = Variable(torch.zeros(1, self.cell.hidden_size)), \
                          Variable(torch.zeros(1, self.cell.hidden_size))

    def next(self, iinput):

        return RNNState(self.cell, self.cell(iinput, self.hidden))

    def __call__(self):
        return self.hidden[0]



class ArcHybridLSTMModel(nn.Module):
    def __init__(self, words, rels, w2i, options):

        super(ArcHybridLSTMModel, self).__init__()
        random.seed(1)

        self.activations = {'tanh': F.tanh, 'sigmoid': F.sigmoid, 'relu': F.relu}
        self.activation = self.activations[options.activation]

        self.oracle = options.oracle
        self.ldims = options.lstm_dims 
        self.wdims = options.wembedding_dims
        self.pdims = options.pembedding_dims
        self.rdims = options.rembedding_dims
        self.layers = options.lstm_layers
        self.wordsCount = words
        self.vocab = {word: ind+3 for word, ind in w2i.iteritems()}
        #self.pos = {word: ind+3 for ind, word in enumerate(pos)}
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.irels = rels

        self.headFlag = options.headFlag
        self.rlMostFlag = options.rlMostFlag
        self.rlFlag = options.rlFlag
        self.k = options.window

        self.num_transitioins = 4

        self.nnvecs = (1 if self.headFlag else 0) + (2 if self.rlFlag or self.rlMostFlag else 0)

        self.external_embedding = None
        if options.external_embedding is not None:
            external_embedding_fp = open(options.external_embedding,'r')
            external_embedding_fp.readline()
            self.external_embedding = {line.split(' ')[0] : [float(f) for f in line.strip().split(' ')[1:]] for line in external_embedding_fp}
            external_embedding_fp.close()

            self.edim = len(self.external_embedding.values()[0])
            self.noextrn = [0.0 for _ in xrange(self.edim)]
            self.extrnd = {word: i + 3 for i, word in enumerate(self.external_embedding)}
            np_emb = np.zeros((len(self.external_embedding) + 3, self.edim))
            for word, i in self.extrnd.iteritems():
                np_emb[i] = self.external_embedding[word]

            self.elookup        = nn.Embedding(*np_emb.shape)
            self.elookup.weight.data.copy_(torch.from_numpy(np_emb))

            self.extrnd['*PAD*'] = 1
            self.extrnd['*INITIAL*'] = 2

            print 'Load external embedding. Vector dimensions', self.edim
            
        dims = self.wdims + self.pdims + (self.edim if self.external_embedding is not None else 0)
        self.blstmFlag = options.blstmFlag
        self.bibiFlag = options.bibiFlag

        if self.bibiFlag:
            self.surfaceBuilders = [nn.LSTMCell(dims, self.ldims, 1), nn.LSTMCell(dims, self.ldims, 1)]
            self.bsurfaceBuilders = [nn.LSTMCell(self.ldims * 2, self.ldims, 1), nn.LSTMCell(self.ldims * 2, self.ldims, 1)]

        elif self.blstmFlag:
            if self.layers > 0:
                self.surfaceBuilders = [nn.LSTMCell(dims, self.ldims, self.layers), nn.LSTMCell(dims, self.ldims, self.layers)]
            else:
                self.surfaceBuilders = [nn.RNNCell(dims, self.ldims, 1), nn.LSTMCell(dims, self.ldims, 1)]

        for i, b in enumerate(self.surfaceBuilders):
            self.add_module('surfaceBuilders%i' % i, b) 
        if hasattr(self, 'bsurfaceBuilders'):
            for i, b in enumerate(self.bsurfaceBuilders):
                self.add_module('bsurfaceBuilders%i' %i, b)

        self.hidden_units  = options.hidden_units
        self.hidden2_units = options.hidden2_units

        self.vocab['*PAD*'] = 1
        self.vocab['*INITIAL*'] = 2

        self.wlookup = nn.Embedding(len(words) + 3, self.wdims)
        self.rlookup = nn.Embedding(len(rels)     , self.rdims)

        self.word2lstm     = Parameter((self.wdims + self.pdims + (self.edim if self.external_embedding is not None else 0), self.ldims * 2))
        self.word2lstmbias = Parameter((self.ldims * 2))
        self.lstm2lstm     = Parameter((self.ldims * self.nnvecs * 2 + self.rdims, self.ldims * 2))
        self.lstm2lstmbias = Parameter((self.ldims * 2))

        self.hidLayer = Parameter((self.ldims * 2 * self.nnvecs * (self.k + 1), self.hidden_units))
        self.hidBias  = Parameter((self.hidden_units))

        if self.hidden2_units:

            self.hid2Layer = Parameter((self.hidden_units, self.hidden2_units))
            self.hid2Bias  = Parameter((self.hidden2_units))

        self.outLayer = Parameter((self.hidden2_units if self.hidden2_units > 0 else self.hidden_units, self.num_transitioins))
        self.outBias  = Parameter((self.num_transitioins))

        self.rhidLayer = Parameter((self.ldims * 2 * self.nnvecs * (self.k + 1), self.hidden_units))
        self.rhidBias  = Parameter((self.hidden_units))

        if self.hidden2_units:
            self.rhid2Layer = Parameter((self.hidden_units, self.hidden2_units))
            self.rhid2Bias  = Paramter((self.hidden2_units))

        self.routLayer = Parameter((self.hidden2_units if self.hidden2_units > 0 else self.hidden_units, 2 * (len(self.irels) + 0) + 1))
        self.routBias  = Parameter((2 * (len(self.irels) + 0) + 1))

    def evaluate(self, stack, buf, train, lstms=None):
        if train:
            topStack = [ lstms[stack.roots[-i-1].lstms] if len(stack) > i else [self.empty] for i in xrange(self.k) ]
            topBuffer = [ lstms[buf.roots[i].lstms] if len(buf) > i else [self.empty] for i in xrange(1) ]
        else:
            topStack = [ stack.roots[-i-1].lstms if len(stack) > i else [self.empty] for i in xrange(self.k) ]
            topBuffer = [ buf.roots[i].lstms if len(buf) > i else [self.empty] for i in xrange(1) ]

        input = cat(list(chain(*(topStack + topBuffer))))

        if self.hidden2_units > 0:
            #mul1 = self.activation(torch.mm(input, self.rhidLayer) + self.rhidBias)
            #mul2 = self.activation(torch.mm(self.activation(torch.mm(input, self.rhidLayer) + self.rhidBias), self.rhid2Layer) + self.rhid2Bias)
            routput = (torch.mm(self.activation(torch.mm(self.activation(torch.mm(input, self.rhidLayer) + self.rhidBias), self.rhid2Layer) + self.rhid2Bias), self.routLayer) + self.routBias)
        else:
            #mul1 = self.activation(torch.mm(input, self.rhidLayer) + self.rhidBias)
            routput = (torch.mm(self.activation(torch.mm(input, self.rhidLayer) + self.rhidBias), self.routLayer) + self.routBias)

        if self.hidden2_units > 0:
            #mul1 = self.activation(torch.mm(input, self.hidLayer) + self.hidBias)
            #mul2 = self.activation(torch.mm(self.activation(torch.mm(input, self.hidLayer) + self.hidBias), self.hid2Layer) + self.hid2Bias)
            output = (torch.mm(self.activation(torch.mm(self.activation(torch.mm(input, self.hidLayer) + self.hidBias), self.hid2Layer) + self.hid2Bias), self.outLayer) + self.outBias)
        else:
            #mul1 = self.activation(torch.mm(input, self.hidLayer) + self.hidBias)
            output = (torch.mm(self.activation(torch.mm(input, self.hidLayer) + self.hidBias), self.outLayer) + self.outBias)

        scrs, uscrs = get_data(routput).numpy()[0], get_data(output).numpy()[0]

        #transition conditions
        left_arc_conditions = len(stack) > 0 and len(buf) > 0
        right_arc_conditions = len(stack) > 1 and stack.roots[-1].id != 0
        shift_conditions = len(buf) > 0 and buf.roots[0].id != 0
        reduce_condition = len(stack) > 0

        uscrs0 = uscrs[0]
        uscrs1 = uscrs[1]
        uscrs2 = uscrs[2]
        uscrs3 = uscrs[3]

        if train:
            output  = output.view(-1)
            routput = routput.view(-1)
            output0 = output[0]
            output1 = output[1]
            output2 = output[2]
            output3 = output[3]
            
            ret = [ [ (rel, 0, scrs[1 + j * 2] + uscrs1, routput[1 + j * 2 ] + output1) for j, rel in enumerate(self.irels) ] if left_arc_conditions else [],
                    [ (rel, 1, scrs[2 + j * 2] + uscrs2, routput[2 + j * 2 ] + output2) for j, rel in enumerate(self.irels) ] if right_arc_conditions else [],
                    [ (None, 2, scrs[0] + uscrs0, routput[0] + output0) ] if shift_conditions else [],
                    [ (None, 3, scrs[0] + uscrs3, routput[0] + output3) ] if reduce_condition else []]
        else:
            s1,r1 = max(zip(scrs[1::2],self.irels))
            s2,r2 = max(zip(scrs[2::2],self.irels))
            s1 += uscrs1
            s2 += uscrs2
            ret = [ [ (r1, 0, s1) ] if left_arc_conditions else [],
                    [ (r2, 1, s2) ] if right_arc_conditions else [],
                    [ (None, 2, scrs[0] + uscrs0) ] if shift_conditions else [],
                    [ (None, 3, scrs[0] + uscrs3) ] if reduce_condition else []]
                   
        return ret
        #return [ [ (rel, 0, scrs[1 + j * 2 + 0] + uscrs[1], routput[1 + j * 2 + 0] + output[1]) for j, rel in enumerate(self.irels) ] if len(stack) > 0 and len(buf) > 0 else [],
        #         [ (rel, 1, scrs[1 + j * 2 + 1] + uscrs[2], routput[1 + j * 2 + 1] + output[2]) for j, rel in enumerate(self.irels) ] if len(stack) > 1 else [],
        #         [ (None, 2, scrs[0] + uscrs[0], routput[0] + output[0]) ] if len(buf) > 0 else [] ]

    def Init(self):
        evec = self.elookup(scalar(1)) if self.external_embedding is not None else None
        paddingWordVec = self.wlookup(scalar(1))
        paddingVec = F.tanh(torch.mm(cat([paddingWordVec, evec]), self.word2lstm)  + self.word2lstmbias)
        self.empty = paddingVec if self.nnvecs == 1 else cat([paddingVec for _ in xrange(self.nnvecs)])

    def getWordEmbeddings(self, sentence, train):
        if train:
            sent_ivec = []

            for root in sentence:
                c = float(self.wordsCount.get(root.norm, 0))
                dropFlag =  not train or (random.random() < (c/(0.25+c)))

                wordvec = self.wlookup(scalar(int(self.vocab.get(root.norm, 0))) if dropFlag else scalar(0))
                #root.posvec = self.plookup(scalar(int(self.pos[root.pos]))) if self.pdims > 0 else None

                if self.external_embedding is not None:
                    
                    if root.form in self.external_embedding:
                        evec = self.elookup(scalar(self.extrnd[root.form]))
                    elif root.norm in self.external_embedding:
                        evec = self.elookup(scalar(self.extrnd[root.norm]))
                    else:
                        evec = self.elookup(scalar(0))
                else:
                    evec = None
                #root.ivec = cat([root.wordvec, root.posvec, root.evec])
                ivec = cat([wordvec, evec])
                sent_ivec.append(ivec)

            if self.blstmFlag:
                forward_cell  = self.surfaceBuilders[0]
                forward_h     = Variable(torch.zeros(1, forward_cell.hidden_size)).cuda(), \
                                Variable(torch.zeros(1, forward_cell.hidden_size)).cuda()
                backward_cell = self.surfaceBuilders[1]
                backward_h    = Variable(torch.zeros(1, backward_cell.hidden_size)).cuda(), \
                              Variable(torch.zeros(1, backward_cell.hidden_size)).cuda()
                
                sent_fvec, sent_bvec = [], []
                for fivec, rivec in zip(sent_ivec, reversed(sent_ivec)):
                    forward_h  = forward_cell(fivec, forward_h)
                    backward_h = backward_cell(rivec, backward_h)
                   
                    sent_fvec.append(forward_h[0])
                    sent_bvec.insert(0, backward_h[0])

                sent_vec = []
                for i in range(len(sent_fvec)):
                    sent_vec.append(cat([sent_fvec[i], sent_bvec[i]]))

                if self.bibiFlag:
                    bforward_cell  = self.bsurfaceBuilders[0]
                    bforward_h     = Variable(torch.zeros(1, bforward_cell.hidden_size)).cuda(), \
                                     Variable(torch.zeros(1, bforward_cell.hidden_size)).cuda()
                    bbackward_cell = self.bsurfaceBuilders[1]
                    bbackward_h    = Variable(torch.zeros(1, bbackward_cell.hidden_size)).cuda(), \
                                  Variable(torch.zeros(1, bbackward_cell.hidden_size)).cuda()
                
                    sent_bfvec, sent_bbvec = [], []    
                    for bfvec, bbvec in zip(sent_vec, reversed(sent_vec)):
                       
                        bforward_h  = bforward_cell(bfvec, bforward_h)
                        bbackward_h = bbackward_cell(bbvec, bbackward_h)
                        sent_bfvec.append(bforward_h[0])
                        sent_bbvec.insert(0, bbackward_h[0])

                    sent_vec = []
                    for i in range(len(sent_bfvec)):
                        sent_vec.append(cat([sent_bfvec[i], sent_bbvec[i]]))

            else:
                sent_vec = []
                for ivec in sent_ivec:
                    ivec = (torch.mm(ivec, self.word2lstm) + self.word2lstmbias)
                    sent_vec.append(F.tanh(ivec))

            return sent_vec

        else:
            for root in sentence:
                c = float(self.wordsCount.get(root.norm, 0))
                dropFlag =  not train or (random.random() < (c/(0.25+c)))
            
                root.wordvec = self.wlookup(scalar(int(self.vocab.get(root.norm, 0))) if dropFlag else scalar(0))
                #root.posvec = self.plookup(scalar(int(self.pos[root.pos]))) if self.pdims > 0 else None

                if self.external_embedding is not None:
                    if root.form in self.external_embedding:
                        root.evec = self.elookup(scalar(self.extrnd[root.form]))
                    elif root.norm in self.external_embedding:
                        root.evec = self.elookup(scalar(self.extrnd[root.norm]))
                    else:
                        root.evec = self.elookup(scalar(0))
                else:
                    root.evec = None
                #root.ivec = cat([root.wordvec, root.posvec, root.evec])
                root.ivec = cat([root.wordvec, root.evec])


            if self.blstmFlag:
                forward  = RNNState(self.surfaceBuilders[0])
                backward = RNNState(self.surfaceBuilders[1])

                for froot, rroot in zip(sentence, reversed(sentence)):
                    
                    forward = forward.next(froot.ivec)
                    backward = backward.next(rroot.ivec)
                    froot.fvec = forward()
                    rroot.bvec = backward()
                for root in sentence:
                    root.vec = cat([root.fvec, root.bvec])

                if self.bibiFlag:
                    bforward  = RNNState(self.bsurfaceBuilders[0])
                    bbackward = RNNState(self.bsurfaceBuilders[1])

                    for froot, rroot in zip(sentence, reversed(sentence)):
                       
                        bforward = bforward.next(froot.vec)
                        bbackward = bbackward.next(rroot.vec)
                        froot.bfvec = bforward()
                        rroot.bbvec = bbackward()

                    for root in sentence:
                        root.vec = cat([root.bfvec, root.bbvec])

            else:
                for root in sentence:
                    root.ivec = (torch.mm(root.ivec, self.word2lstm) + self.word2lstmbias)
                    root.vec = F.tanh(root.ivec)

    def Predict(self, sentence):     
        conll_sentence = sentence[1:] + [sentence[0]]
        self.getWordEmbeddings(conll_sentence, False)
        stack = ParseForest([])
        buf = ParseForest(conll_sentence)

        for root in conll_sentence:
            root.lstms = [root.vec for _ in xrange(self.nnvecs)]

        hoffset = 1 if self.headFlag else 0

        while not (len(buf) == 1 and len(stack) == 0):
            scores = self.evaluate(stack, buf, False)
            best = max(chain(*scores), key = itemgetter(2) )

            if best[1] == 2:  # shift
                stack.roots.append(buf.roots[0])
                del buf.roots[0]

            elif best[1] == 0: #left
                child = stack.roots.pop()
                parent = buf.roots[0]

                child.pred_parent_id = parent.id
                child.pred_relation = best[0]

                bestOp = 0
                if self.rlMostFlag:
                    parent.lstms[bestOp + hoffset] = child.lstms[bestOp + hoffset]
                if self.rlFlag:
                    parent.lstms[bestOp + hoffset] = child.vec

            elif best[1] == 1: #right
                child = stack.roots.pop()
                parent = stack.roots[-1]

                child.pred_parent_id = parent.id
                child.pred_relation = best[0]

                bestOp = 1
                if self.rlMostFlag:
                    parent.lstms[bestOp + hoffset] = child.lstms[bestOp + hoffset]
                if self.rlFlag:
                    parent.lstms[bestOp + hoffset] = child.vec

            elif best[1] == 3: # reduce
                child = stack.roots.pop()
                child.pred_parent_id = -1
                child.pred_relation = '_'




def get_optim(opt, parameters):
    if opt == 'sgd':
        return optim.SGD(parameters, lr=opt.lr)
    elif opt == 'adam':
        return optim.Adam(parameters, eps=1e-2)


class ArcHybridLSTM:
    #def __init__(self, words, pos, rels, w2i, options):
    def __init__(self, words, rels, w2i, options):
        #model = ArcHybridLSTMModel(words, pos, rels, w2i, options)
        model = ArcHybridLSTMModel(words, rels, w2i, options)
        self.model = model.cuda() if use_gpu else model
        self.trainer = get_optim('adam', self.model.parameters())


    def Predict(self, conll_path):
        with open(conll_path, 'r') as conllFP:
            for iSentence, sentence in enumerate(read_conll(conllFP, False)):
                self.model.Init()
                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
                self.model.Predict(conll_sentence)
                yield conll_sentence

    def Save(self, fn):
        tmp = fn +'_1e-2_256_100'+ '.tmp'
        torch.save(self.model.state_dict(), tmp)


    def Load(self, fn):
        self.model.load_state_dict(torch.load(fn))
     

    def Train(self, conll_path, epoch):
        mloss = 0.0
        errors = 0
        batch = 0
        eloss = 0.0
        eerrors = 0
        lerrors = 0
        etotal = 0
        ltotal = 0
        ninf = -float('inf')

        hoffset = 1 if self.model.headFlag else 0

        start = time.time()

        fout = open('loss_coco_0001_epoch_%d.log'%epoch, 'w')

        with open(conll_path, 'r') as conllFP:
            shuffledData = list(read_conll(conllFP, True))

            random.shuffle(shuffledData)

            errs = []
            eeloss = 0.0

            self.model.Init()
            non_proj = 0

            for iSentence, sentence in enumerate(shuffledData):
                
                isProj = True 
                if iSentence % 100 == 0 and iSentence != 0:
                    print 'Processing sentence number:', iSentence, 'Loss:', eloss / etotal, 'Errors:', (float(eerrors)) / etotal, 'Labeled Errors:', (float(lerrors) / etotal) , 'Time', time.time()-start
                    #print "check"
                    fout.write(str(eloss/etotal)+'\n')
                    start = time.time()
                    del eerrors, eloss, etotal, lerrors, ltotal
                    eerrors = 0
                    eloss = 0.0
                    etotal = 0
                    lerrors = 0
                    ltotal = 0

                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
                conll_sentence = conll_sentence[1:] + [conll_sentence[0]]
                if iSentence != 0:
                    del sent_vec, lstms

                sent_vec = self.model.getWordEmbeddings(conll_sentence, True)
                lstms = []
                
                stack = ParseForest([])
                buf = ParseForest(conll_sentence)

                for i in range(len(sent_vec)):
                    buf.roots[i].lstms = i
                    lstms.append([sent_vec[i] for _ in xrange(self.model.nnvecs)])

                hoffset = 1 if self.model.headFlag else 0

                while not (len(buf) == 1 and len(stack) == 0):

                    scores = self.model.evaluate(stack, buf, True, lstms)
                    scores.append([(None, self.model.num_transitioins, ninf ,None)])

                    alpha = stack.roots[:-2] if len(stack) > 2 else []
                    s1 = [stack.roots[-2]] if len(stack) > 1 else []
                    s0 = [stack.roots[-1]] if len(stack) > 0 else []
                    b = [buf.roots[0]] if len(buf) > 0 else []
                    beta = buf.roots[1:] if len(buf) > 1 else []


                    left_cost   = ( len([h for h in s1 + beta if h.id == s0[0].parent_id]) +
                                    len([d for d in b + beta if d.parent_id == s0[0].id]) ) if len(scores[0]) > 0 else 1
                    right_cost  = ( len([h for h in b + beta if h.id == s0[0].parent_id]) +
                                    len([d for d in b + beta if d.parent_id == s0[0].id]) )  if len(scores[1]) > 0 else 1
                    shift_cost  = ( len([h for h in s1 + alpha if h.id == b[0].parent_id]) +
                                    len([d for d in s0 + s1 + alpha if d.parent_id == b[0].id]) )  if len(scores[2]) > 0 else 1
                    reduce_cost = ( len([h for h in s1 + b + beta if h.id == s0[0].parent_id]) + 
                                    len([d for d in b + beta if d.parent_id == s0[0].id]) ) if len(scores[3]) > 0 else 1

                    #print "\nWord in Buff: "
                    #for word in b+beta:
                    #    print "buf parent id:  ", word.parent_id
                    #    print "word id:  ", word.id
                    #    print "word:  ", word.form
                    

                    if len(stack) > 0:
                        #print "parent id:  ",  s0[0].parent_id
                        if s0[0].parent_id == -1 and reduce_cost == 0:
                            left_cost  += 1
                            right_cost += 1
                            shift_cost += 1

                    costs = (left_cost, right_cost, shift_cost, reduce_cost, 1)

                    #print "CCCOST   ", costs 
                    
                    #costs = (left_cost, right_cost, shift_cost, reduce_cost, isObj_cost, isPred_cost, 1)
                    
                    try: 
                        bestValid = max(( s for s in chain(*scores) if costs[s[1]] == 0 and ( s[1] == 2 or  s[0] == stack.roots[-1].relation or s[0] == None) ), key=itemgetter(2))
                    except:
                        print "length of stack: ", len(stack.roots)
                        #for roots in stack.roots:
                        #    print roots.parent
                        #    print roots.children
                        print "This is non projective"
                        exit()
                        non_proj += 1
                        isProj = False
                        break

                    bestWrong = max(( s for s in chain(*scores) if costs[s[1]] != 0 or  ( s[1] != 2 and s[0] != stack.roots[-1].relation ) ), key=itemgetter(2))
                    #best = bestValid if ( (not self.model.oracle) or (bestValid[2] - bestWrong[2] > 1.0) or (bestValid[2] > bestWrong[2] and random.random() > 0.1) ) else bestWrong
                    best = bestValid
                    #print "No. %d sentences, best[1] = %d" % (iSentence, best[1])
                   
                    if best[1] == 2:
                        stack.roots.append(buf.roots[0])
                        del buf.roots[0]

                    elif best[1] == 0:
                        child = stack.roots.pop()
                        parent = buf.roots[0]

                        child.pred_parent_id = parent.id
                        child.pred_relation = best[0]

                        bestOp = 0
                        if self.model.rlMostFlag:
                            lstms[parent.lstms][bestOp + hoffset] = lstms[child.lstms][bestOp + hoffset]
                        if self.model.rlFlag:
                            lstms[parent.lstms][bestOp + hoffset] = sent_vec[child.lstms]


                    elif best[1] == 1:
                        child = stack.roots.pop()
                        parent = stack.roots[-1]

                        child.pred_parent_id = parent.id
                        child.pred_relation = best[0]

                        bestOp = 1
                        if self.model.rlMostFlag:
                            lstms[parent.lstms][bestOp + hoffset] = lstms[child.lstms][bestOp + hoffset]
                        if self.model.rlFlag:
                            lstms[parent.lstms][bestOp + hoffset] = sent_vec[child.lstms]

                    elif best[1] == 3:
                        child = stack.roots.pop()
                        child.pred_parent_id = -1
                        child.pred_relation = '_'

                        bestOp = 3
                        
                    if bestValid[2] < bestWrong[2] + 1.0:
                        loss = bestWrong[3] - bestValid[3]
                        mloss += 1.0 + bestWrong[2] - bestValid[2]
                        eloss += 1.0 + bestWrong[2] - bestValid[2]
                        errs.append(loss)

                        del loss

                    if best[1] != 2 and (child.pred_parent_id != child.parent_id or child.pred_relation != child.relation):
                        lerrors += 1
                        if child.pred_parent_id != child.parent_id:
                            errors += 1
                            eerrors += 1

                    etotal += 1
                    del scores

                #print "Finish %d sentences" %(iSentence)
                #print os.system('nvidia-smi')
                #exit()
               
                if len(errs) > 50: # or True:
                    eerrs = torch.sum(cat(errs))
                    scalar_loss = get_data(eerrs).numpy()[0]
                    eerrs.backward()
                    self.trainer.step()
                    del eerrs
                    errs = []
                    lerrs = []

                    self.model.Init()

                self.trainer.zero_grad()


        if len(errs) > 0:
            eerrs = torch.sum(cat(errs)) # * (1.0/(float(len(errs))))
            get_data(eerrs).numpy()[0]
            eerrs.backward()
            self.trainer.step()

            del eerrs
            errs = []
            lerrs = []

        self.trainer.zero_grad()
        print "Loss: ", mloss/iSentence

        #del mloss






