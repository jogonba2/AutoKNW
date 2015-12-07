#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Config
import logging
from re import sub

def delete_duplicates(s):      
	sn = ""
	for i in xrange(len(s)):
		if i==0: sn += s[i]
		else:    sn += s[i] if s[i]!=sn[-1] else ""
	return sn

def remove_non_alfanumeric_1(sentence): return sub(u"[^\wáéíóúñçàèìòù]",u" ",sentence)

def remove_non_alfanumeric_2(sentence,non_alfanumeric): 
	for non_alfanumeric in non_alfanumeric:
		sentence = sentence.replace(non_alfanumeric,"")
	return sentence

def tokenizer(sentence,pattern_tokenize,lowercase): 
	if lowercase==True: return [word.strip().lower() for word in sentence.split(pattern_tokenize) if word!=""]
	else: return [word.strip() for word in sentence.split(pattern_tokenize) if word]

def remove_stopwords(tokens,stopwords):     return [word for word in tokens if word not in stopwords if word!=""]

def stemming(tokens,stemmer):     return [stemmer.stem(word) for word in tokens]

def lemmatize(tokens,lemmatizer):   return [lemmatizer.lemmatize(word,pos='v') for word in tokens]

def split_sentence(s): return s.split(" ")

def get_ngrams(tokens,n): return list(set([" ".join(tokens[i:i+n]) for i in xrange(len(tokens)-n+1)]))

def normalize(sentence,lowercase=True,make_stemming=False,make_lemmatize=False,remove_duplicates=True,remove_stopw=True,remove_alfanumeric=True,ngrams=1,non_alfanumeric_func=Config.NON_ALFANUMERIC_FUNC,stemmer=Config.STEMMER,lemmatizer=Config.LEMMATIZER,pattern_tokenize=Config.PATTERN_TOKENIZER,stopwords=Config.STOPWORDS,non_alfanumeric_patterns=Config.NON_ALFANUMERIC_PATTERNS): 
	# Tokenize #
	tokens = tokenizer(sentence,pattern_tokenize,lowercase)
	# Remove alfanumeric #
	if remove_alfanumeric==True: 
		if non_alfanumeric_func==1: tokens = [remove_non_alfanumeric_1(token) for token in tokens]
		if non_alfanumeric_func==2: tokens = [remove_non_alfanumeric_2(token,non_alfanumeric_patterns) for token in tokens]
	# Remove duplicate letters #
	if remove_duplicates==True:  tokens = [delete_duplicates(token) for token in tokens]
	# Stopwords #
	if remove_stopw==True:      tokens = remove_stopwords(tokens,stopwords)
	# Lemmatize #
	if make_lemmatize==True:    tokens = lemmatize(tokens,lemmatizer)
	# Stemming  #
	if make_stemming==True:     tokens = stemming(tokens,stemmer)
	# Get ngrams #
	if ngrams!=1: 		  tokens = get_ngrams(tokens,ngrams)

	return tokens

def normalize_sentences(sentences,lowercase=True,make_stemming=False,make_lemmatize=False,remove_duplicates=True,remove_stopw=True,remove_alfanumeric=True,ngrams=1,non_alfanumeric_func=Config.NON_ALFANUMERIC_FUNC,stemmer=Config.STEMMER,lemmatizer=Config.LEMMATIZER,pattern_tokenize=Config.PATTERN_TOKENIZER,stopwords=Config.STOPWORDS,non_alfanumeric_patterns=Config.NON_ALFANUMERIC_PATTERNS): 
	res = []
	for (sentence,cat) in sentences:
		res.append((normalize(sentence,lowercase,make_stemming,make_lemmatize,remove_duplicates,remove_stopw,remove_alfanumeric,ngrams,non_alfanumeric_func,stemmer,lemmatizer,pattern_tokenize,stopwords,non_alfanumeric_patterns),cat))
	return res
