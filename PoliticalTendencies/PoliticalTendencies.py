#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '../')
from Kernel import Config
from Kernel import Preprocess
from Kernel import Classifier
from Kernel import FeatureExtractor
from Kernel import Optimizer
from Kernel import Statistics
from Kernel import Utils
from gensim.models import Word2Vec
import logging

CORPUS = "./corpus_political_tendencies/"

if Config.VERBOSE: logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) 

## SAVE MODEL ##
sentences = Utils.get_sentences(CORPUS)
sentences = Preprocess.normalize_sentences(sentences,True,False,False,False,False,True,1)
model     = Word2Vec([sentence for (sentence,cat) in sentences],min_count=Config.MIN_COUNT_W2V,size=Config.SIZE_W2V,window=Config.WINDOW_W2V)
sentences = FeatureExtractor.get_sentences_representation(sentences,model,Config.SENTENCE_REPRESENTATION)
Utils.serialize("dest.m",model)
Utils.serialize("sentences.m",sentences)
################

## TESTING PROJECT CLASSIFIER ##
model = Utils.unserialize("dest.m")[0]
sentences = Utils.unserialize("sentences.m")[0]
# Add logic to apply analysis only to politic sentences #
sentence = "A falta de 16 escaños por asignar, la Mesa de Unidad Democrática logra 103 asientos, mayoría suficiente para aprobar leyes habilitantes"
sentence = Preprocess.normalize(sentence,True,False,False,False,False,True,1)
sentence = Config.SENTENCE_REPRESENTATION(sentence,model)
print Classifier.get_class(sentence,sentences,1)
# Get more than 1 class #
print Classifier.get_k_classes(sentence,sentences,2)
################################

## STATISTICS ##
#sentences = Utils.get_sentences(CORPUS)
#sentences = Preprocess.normalize_sentences(sentences,True,False,False,False,False,True,1)
#Statistics.leaving_one_out_nearest(sentences,1)
#Statistics.leaving_one_out_supervised_classifier(sentences,Config.CLASSIFIER)
################

## OPTIMIZE PARAMETERS ##
		# ... #
#########################
