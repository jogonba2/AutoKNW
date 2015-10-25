#!/usr/bin/env python
# -*- coding: utf-8 -*-

from re import sub
from config import *

""" Refactorizar """

class Tokenizer(object):
	
	def __init__(self,configuration):
		self.configuration = configuration
	
	# ... #
	
# Elimina stopwords del documento #
def remove_stopwords(document): return [w for w in document if w.lower() not in STOPWORDS]

# Stemming al documento #
def make_stemming(document):    return [STEMMER.stem(word) for word in document]

# Elimina caracteres no alfanuméricos menos letras acentuadas y caracteres especiales ç y ñ #
def remove_non_alpha(document): return sub(u"[^\wáéíóúñçàèìòù]",u" ",document)

# Wrapper de split #
def split_by_separator(document,separator): return document.split(separator)

# Eliminar caracteres nulos y pasa a minusculas los terminos de la lista #
def process_document(document): return [term.lower() for term in document if term!=""]


