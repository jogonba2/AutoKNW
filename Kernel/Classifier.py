#!/usr/bin/env python
# -*- coding: utf-8 -*-

import Config
from math import log
from numpy import ndarray

def distance_representations(s1,s2): 
	try:
		d = Config.DISTANCE_CLASSIFICATION(s1,s2)
		return log(d) if d>0 else d
	except: return float("inf")

def get_k_most_similar_sentences(s1,sentences,k):
	k_similar_sentences = []
	for i in xrange(len(sentences)):
		sentence 		  = sentences[i][0]
		cat				  = sentences[i][1]
		if type(s1)==ndarray and type(sentence)==ndarray:
			m = distance_representations(s1,sentence)
			if len(k_similar_sentences)!=k: k_similar_sentences.append((m,sentence,cat))
			else:
				max_k_similar_sentences = max(k_similar_sentences)
				if m<max_k_similar_sentences[0]:
					k_similar_sentences[k_similar_sentences.index(max_k_similar_sentences)] = ((m,sentence,cat))
	return k_similar_sentences

def get_class(s1,sentences,k):
	k_similar_sentences = get_k_most_similar_sentences(s1,sentences,k)
	h,m,c_class = {},0,-1
	if not k_similar_sentences: return None
	else:
		for (distance,sentence,cat) in k_similar_sentences:
			if cat not in h: h[cat]  = 1
			else:			 h[cat] += 1
			if h[cat]>m: m,c_class = h[cat],cat
		return c_class
		
def get_k_classes(s1,sentences,k):
	k_similar_sentences = get_k_most_similar_sentences(s1,sentences,k)
	h = {}
	if not k_similar_sentences: return None
	else:
		for (distance,sentence,cat) in k_similar_sentences:
			if cat not in h: h[cat]  = 1
			else:			 h[cat] += 1
	l = h.items()
	l.sort(key=lambda x:x[1])
	return l[-k:len(l)]
	return c_class
