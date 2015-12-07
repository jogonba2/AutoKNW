#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import Config
from gensim.models import Word2Vec
from Classifier import get_class
from FeatureExtractor import get_sentences_representation
from warnings import filterwarnings

# Filter warnings from other libs #
filterwarnings("ignore")
###################################

def leaving_one_out_nearest(sentences,k):
	if Config.VERBOSE: logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) 
	act_index,act_test_sample,err = 0,None,0
	while act_index<len(sentences):
		train_sentences     = [sentences[i] for i in xrange(len(sentences)) if act_index!=i]
		model = Word2Vec([sentence for (sentence,cat) in train_sentences],min_count=Config.MIN_COUNT_W2V,size=Config.SIZE_W2V,window=Config.WINDOW_W2V)
		test_sentence = sentences[act_index]
		test_sentence = (Config.SENTENCE_REPRESENTATION(test_sentence[0],model),test_sentence[1])
		train_sentences_rep = get_sentences_representation(train_sentences,model,Config.SENTENCE_REPRESENTATION)
		if Config.REGULARIZATION: pass # Implement regularization #
		if Config.VERBOSE: logging.info("Testing test sentence: "+str(act_index)+" against "+str(len(train_sentences))+" sentences")
		c_class = get_class(test_sentence[0],train_sentences_rep,k)
		if c_class!=test_sentence[1]: err += 1
		act_index += 1
	print "Accuracy = ",1.0-(float(err)/len(sentences))


def leaving_one_out_supervised_classifier(sentences,clf):
	if Config.VERBOSE: logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) 
	act_index,act_test_sample,err = 0,None,0
	while act_index<len(sentences):
		train_sentences     = [sentences[i] for i in xrange(len(sentences)) if act_index!=i]
		model = Word2Vec([sentence for (sentence,cat) in train_sentences],min_count=Config.MIN_COUNT_W2V,size=Config.SIZE_W2V,window=Config.WINDOW_W2V)
		test_sentence = sentences[act_index]
		test_sentence = (Config.SENTENCE_REPRESENTATION(test_sentence[0],model),test_sentence[1])
		train_sentences_rep = get_sentences_representation(train_sentences,model,Config.SENTENCE_REPRESENTATION)
		if Config.REGULARIZATION: pass # Implement regularization #
		if Config.VERBOSE: logging.info("Testing test sentence: "+str(act_index)+" against "+str(len(train_sentences))+" sentences")
		""" Training """
		h,s,c = {},set([cat for (sentence,cat) in train_sentences_rep]),0
		for cat in s: h[cat] = c; c += 1
		X,Y = [],[]
		for (train_sentence,cat) in train_sentences_rep: 
			X.append(train_sentence); 
			Y.append(h[cat])
		clf.fit(X,Y)
		""" Testing """
		y = clf.predict(test_sentence[0])
		if y[0]!=h[test_sentence[1]]: err += 1
		act_index += 1
	print "Accuracy = ",1.0-(float(err)/len(sentences))



