#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

class Config(object):
	
	def __init__(self,language,flag_stemming,flag_stopword):
		self.__language  = language
		self.__flag_stemming   = False
		self.__flag_stopwords = False
		self.__stopwords = []
		self.__stemmer   = None
		self.__generate_stopwords()
		self.__generate_stemmer()
	
	def _get_language(self):  		return self.__language
	def _get_stopwords(self): 	    return self.__stopwords
	def _get_stemmer(self):   	    return self.__stemmer
	def _get_flag_stopwords(self):  return self.language
	def _get_flag_stemming(self):   return self.flag_stemming
	
	def _set_language(self,language):             self.language       = language
	def _set_flag_stemming(self,flag_stemming):   self.flag_stemming  = flag_stemming
	def _set_flag_stopwords(self,flag_stopwords): self.flag_stopwords = flag_stopwords
	def _set_stopwords(self,stopwords):			  self.stopwords      = stopwords
	def _set_stemmer(self,stemmer):			      self.stemmer	      = stemmer
	
	def __generate_stopwords(self): 
		if self.flag_stopwords:   self.stopwords = [x for x in stopwords.words(language)]
	def __generate_stemmer(self):   
		if self.flag_stemming:    self.stemmer   = SnowballStemmer(language)

""" Eliminar y actualizar tokenizer.py """
language  = "spanish"
flag_stemming   = False
flag_stopwords = False
STOPWORDS = []
STEMMER   = None
if flag_stopwords: STOPWORDS = [x for x in stopwords.words(language)]
if flag_stemming:    STEMMER   = SnowballStemmer(language)
