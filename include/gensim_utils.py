#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gensim import corpora,models,similarities
from tokenizer import tokenize
from os import listdir

""" Clase para manejar las consultas (ACTUALIZAR) _similarity_query_against_corpus, _more_similar_document, ... """
class CorpusQueries:
	pass
	
""" Clase para manejar transformaciones del corpus a otras representaciones (entrenamiento) """

class CorpusTransformation(object):
	
	def __init__(self,corpus,dictionary):
		self.__corpus       	= corpus
		self.__corpus_tfidf 	= None
		self.__dictionary   	= dictionary
		self.__tfidf_model  	= None
		self._set_tfidf_model() # Configuracion automatica el corpus en tfidf (espacio dominio de muchas transformaciones)
		self.__lsi_model        = None
		self.__lda_model        = None
		self.__rp_model         = None
		self.__hdp_model        = None
		self.__similarity_matrix = None
		
	""" Getters """
	def _get_tfidf_model(self): return self.__tfidf_model
	def _get_lsi_model(self):   return self.__lsi_model
	def _get_lda_model(self):   return self.__lda_model
	def _get_rp_model(self):    return self.__rp_model
	def _get_hdp_model(self):   return self.__hdp_model
	def _get_similarity_matrix(self): return self.__similarity_matrix
	
	""" Setters (para entrenamiento online, llamar _add_documents_to_model y settear el nuevo modelo) """
	
	def _set_tfidf_model(self,tfidf_model): self.__tfidf_model = tfidf_model
	def _set_lsi_model(self,lsi_model):		self.__lsi_model   = lsi_model
	def _set_lda_model(self,lda_model):     self.__lda_model   = lda_model
	def _set_rp_model(self,rp_model):       self.__rp_model    = rp_model
	def _set_hdp_model(self,hdp_model):     self.__hdp_model   = hdp_model
	
	""" Configura la representación en espacio [0,1] frecuecia de término * frecuencia inversa de documento (TFIDF) - requiere corpus bag of words -"""  
	def _set_tfidf_model(self): self.__tfidf_model = models.TfidfModel(self.__corpus,normalize=True) if self.__tfidf_model==None else self.__tfidf_model
	
	""" Configura la representación de un espacio latente N-dimensional (LSI) - corpus bag of words o tfidf -"""
	def _set_lsi_model(self,topics,tfidf_corpus=None): 
		if tfidf_corpus==None: self.__lsi_model     = models.LsiModel(self.__corpus, id2word=self.__dictionary, num_topics=topics) if self.__lsi_model==None else self.__lsi_model
		else:                  self.__lsi_model     = models.LsiModel(tfidf_corpus, id2word=self.__dictionary, num_topics=topics) if self.__lsi_model==None else self.__lsi_model

	""" Configura una aproximación a la representación TFIDF para reducir dimensionalidad y cómputo - requiere corpus en tfidf -"""
	def _set_rp_model(self,topics,tfidf_corpus): self.__rp_model     = models.RpModel(tfidf_corpus, num_topics=topics) if self.__rp_model==None else self.__rp_model
	
	""" Configura la representación en un espacio [0,1] con dimensionalidad reducida mediante LDA - requiere corpus bag_of_words -"""
	def _set_lda_model(self,topics): self.__lda_model     = models.LdaModel(self.__corpus,id2word=self.__dictionary,num_topics=topics) if self.__lda_model==None else self.__lda_model
	
	""" Configura la representación HDP (usar con cuidado según la documentación por ser método nuevo) """
	def _set_hdp_model(self): self.__hdp_model     = models.HdpModel(self.__corpus,id2word=self.__dictionary) if self.__hdp_model==None else self.__hdp_model
	
	""" Configura la matriz de similaridad para realizar consultas """
	def _set_similarity_matrix(self,model,corpus): self.__similarity_matrix = similarities.MatrixSimilarity(model[corpus])
	
	""" Usa el modelo actual para transformar un vector o corpus con una representación bag-of-word a la representación del modelo """
	def _apply_transform_to_vector(self,model,data): return [vector for vector in model[data]]
	
	""" Obtener similaridad de un documento respecto a un corpus entero (no tienen porque ser los de entrenamiento,pero se usaran estos) """
	def _similarity_query_against_corpus(self,vector): return sorted(enumerate(self.__similarity_matrix[vector]), key=lambda item: -item[1])
	
	""" Obtener el documento al que más se parece un documento dado """
	def _more_similar_document(self,vector): return self._similarity_query_against_corpus(vector)[0]
	
	""" Añade más documentos al modelo (entrenamiento online solo para modelo LSI) """
	def _add_documents_to_model(self,model,corpus): model.add_documents(corpus); return model
	
	""" Salva el modelo indicado """
	def _save_model(self,model,fd): model.save(fd)
	
	""" Carga el modelo indicado ... """
	def _load_model(self,model,fd): pass
	
	
""" Wrapper para manejar corpus memory-friendly """
class CorpusWrapper(object):
	
	def __init__(self,train_corpus_path,test_corpus_path):
		self.__train_corpus_path = train_corpus_path	
		self.__test_corpus_path  = test_corpus_path
		self.__set_dictionary()
		self.__test_corpus_data  = [vector for vector in self.__iter__() if vector]
	
	""" Metodos setter """
	def _set_train_corpus_path(self,path): self.__train_corpus_path = path
	def _set_test_corpus_path(self,path): self.__test_corpus_path = path
	def _set_dictionary(self,dictionary): self.__dictionary = dictionary
	
	""" Metodos getter """
	def _get_train_corpus_path(self): return self.__train_corpus_path
	def _get_test_corpus_path(self): return self.__test_corpus_path
	def _get_test_corpus_data(self): return self.__test_corpus_data
	def _get_dictionary(self):  return self.__dictionary
	
	""" Construir diccionario a partir de los documentos """
	def __set_dictionary(self):
		self.__dictionary = corpora.Dictionary(tokenize(open(self.__train_corpus_path+"/"+document).read()) 
											   for document in listdir(self.__train_corpus_path))
		self.__dictionary.compactify()
	
	""" Guardar la representación bag-of-words (de momento requiere almacenar todos los documentos en memoria - arreglar -) """
	def _save_bag_of_words(self,fd):
		corpus = [vector for vector in self.__iter__()]
		corpora.MmCorpus.serialize(fd, corpus)	
	
	""" Guardar el diccionario en memoria """
	def _save_dictionary(self,fd): self.__dictionary.save(fd)
	
	""" Cargar diccionario (HACER) """
	def _load_dictionary(self,fd): pass
	""" Cargar corpus (HACER) """
	def _load_corpus(self,fd): pass
	""" Retornar la representacion bag-of-words de un documento dado """
	def _get_bag_of_words(self,doc): return self.__dictionary.doc2bow(tokenize(doc))
	
	"""Retornar la representación bag-of-words (vector-disperso de pares (id,nºocurrencias) por documento) empleando 
	   one-document-at-time - memory-friendly - """
	def __iter__(self):
		for document in sorted(listdir(self.__test_corpus_path)):
			fd = open(self.__test_corpus_path+"/"+document)
			yield self.__dictionary.doc2bow(tokenize(fd.read()))
			
