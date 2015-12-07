#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numpy import cov,array
from numpy import zeros,ones
import Config

def get_sentence_representation_sum(s,model): # VECTOR = SUMA(VECTOR ANTERIOR,VECTOR SIG PALABRA)
	try:     res = zeros(Config.SIZE_W2V)+model[s[0]]
	except:  res = zeros(Config.SIZE_W2V)
	for i in xrange(1,len(s)):
		try: 	res += model[s[i]]
		except: res = res
	return res

def get_sentence_representation_mult(s,model): # VECTOR = PRODUCTO(VECTOR ANTERIOR,VECTOR SIG PALABRA)
	try:     res = ones(Config.SIZE_W2V)*model[s[0]]
	except:  res = ones(Config.SIZE_W2V)
	for i in xrange(1,len(s)):
		try: 	res *= model[s[i]]
		except: res = res
	return res

def get_sentence_representation_distance(s,model): # VECTOR = DISTANCIA(VECTOR ANTERIOR,VECTOR SIG PALABRA)
	try:     res = 0+model[s[0]]
	except:  res = zeros(Config.SIZE_W2V);
	for i in xrange(1,len(s)):
		try: 	
			aux = model[s[i]]
			if type(aux)==int: aux = zeros(Config.SIZE_W2V)
			res += Config.DISTANCE_REPRESENTATION(res,aux)
		except: res = res
	return res

def get_sentence_representation_weighted(s,model,weights): # SUMA PONDERADA FINAL#
	try:     res = model[s[0]]
	except:  res = zeros(Config.SIZE_W2V)
	for i in xrange(1,len(s)):
		try: 	res += model[s[i]]
		except: res = res
	return sum([res[i]*weights[i] for i in xrange(len(res))])

def get_sentence_representation_covariance(s,model): # VECTOR = SUMA DE LAS FILAS NORMALIZADA DE LA MATRIZ DE COVARIANZA DE M=MATRIZ FORMADA POR LOS VECTORES DE PALABRAS QUE LO COMPONEN #
	try:     res = [model[s[0]]]
	except:  res = []
	for i in xrange(1,len(s)):
		try: 	res.append(model[s[i]])
		except: res = res
	return cov(array(res))/len(s)

def get_sentences_representation(sentences,model,frepresentation): return [(frepresentation(sentence,model),cat) for (sentence,cat) in sentences]
