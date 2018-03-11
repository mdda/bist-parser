# PyTorch Implementation of Transition based BIST Parser

#### Requirement
 - Python 2.7
 - PyTorch 0.3

#### Data Format
The software requires `training.conll` and `development.conll` files formatted according to the [CoNLL data format](https://ilk.uvt.nl/conll/#dataformat), or a `training.conllu` and `development.conllu` files formatted according to the [CoNLLU data format](http://universaldependencies.org/format.html).

In order to reproduce the results in [Simple and Accurate Dependency Parsing
Using Bidirectional LSTM Feature Representations](https://www.transacl.org/ojs/index.php/tacl/article/viewFile/885/198), `training.conll` and `development.conll` can be generated from standard Penn Treebank dataset (Standford Dependencies).

#### Training
```
$ cd /barchybrid_torch/barchybrid/
$ python src/parser.py --outdir ./output --train training.conll --dev development.conll  --epochs 30 --lstmdims 125 --lstmlayers 2 [--extrn extrn.vectors] --bibi-lstm --k 3 --usehead --userl
```
--extrn is optional, for people who would like to use pretrained word embedding.

extrn.vectors is the same word embeddings used in [Transition-Based Dependency Parsing with Stack Long Short-Term Memory](https://arxiv.org/abs/1505.08075) which can be downloaded from the authors [github repository](https://github.com/clab/lstm-parser/) and [here](https://drive.google.com/file/d/0B8nESzOdPhLsdWF2S1Ayb1RkTXc/view).

#### Testing

The command for parsing a `test.conll` file formatted according to the [CoNLL data format](https://ilk.uvt.nl/conll/#dataformat) with a previously trained model is:

```
python src/parser.py --predict --outdir [results directory] --test test.conll [--extrn extrn.vectors] --model [trained model file] --params [param file generate during training]
```

