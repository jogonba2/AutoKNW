#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gensim import corpora,models,similarities
from tokenizer import tokenize
from os import listdir

""" Clase para manejar transformaciones del corpus a otras representaciones (entrenamiento) """

class CorpusTransformation(object):
	
	def __init__(self,corpus,dictionary):
		self.__corpus      = corpus
		self.__dictionary  = dictionary
		self.__tfidf_model = None
		self._set_tfidf_model() # Configurar automatica el corpus en tfidf (espacio dominio de muchas transformaciones)
		self.__lsi_model   = None
		# Implementar los modelos que quedan #
		self.__lda_model   = None
		self.__rp_model    = None
		self.__hdp_model   = None
		
	""" Getters """
	def _get_tfidf_model(self): return self.__tfidf_model
	def _get_lsi_model(self):   return self.__lsi_model
	
	""" Setters (para entrenamiento online, llamar _add_documents_to_model y settear el nuevo modelo) """
	
	def _set_tfidf_model(self,tfidf_model): self.__tfidf_model = tfidf_model
	def _set_lsi_model(self,lsi_model):		self.__lsi_model   = lsi_model
	
	""" Configura la representación en espacio [0,1] frecuecia de término * frecuencia inversa de documento (TFIDF) """  
	def _set_tfidf_model(self): self.__tfidf_model = models.TfidfModel(self.__corpus,normalize=True) if self.__tfidf_model==None else self.__tfidf_model
	
	""" Configura la representación de un espacio latente N-dimensional (LSI) """
	def _set_lsi_model(self,topics): self.__lsi_model     = models.LsiModel(self.__corpus, id2word=self.__dictionary, num_topics=topics) if self.__lsi_model==None else self.__lsi_model
		
		
	""" Usa el modelo actual para transformar un vector o corpus con una representación bag-of-word a la representación del modelo """
	def _apply_transform_to_vector(self,model,data): return [vector for vector in model[data]]
	
	""" Extrae el tema con el que más similaridad tiene un documento (en bag of words) """
	def _more_similar_topic_for_document(self,model_representation): 
		return max(model_representation,key=lambda t:t[1])[0]

	""" Extra el tema con el que más similaridad tiene para un conjunto de documentos (en bag of words) """
	def _more_similar_topic_for_documents(self,model_representations): return [self._more_similar_topic_for_document(model_representation) for model_representation in model_representations]
	
	""" Añade más documentos al modelo (entrenamiento online) """
	def _add_documents_to_model(self,model,corpus): model.add_documents(corpus); return model
	""" Salva el modelo indicado """
	def _save_model(self,model,fd): model.save(fd)
	
	""" Carga el modelo indicado ... """
	def _load_model(self,model,fd): pass
	
	
""" Wrapper para manejar corpus memory-friendly """
## Dictionary es train_corpus_data!!! ##
## test_corpus_data es la representacion bag of words!! ##
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
	
	""" Guardar la representación bag-of-words (de momento requiere almacenar todos los documentos en memoria - arreglar -)"""
	def _save_bag_of_words(fd):
		corpus = [vector for vector in self.__iter__()]
		corpora.MmCorpus.serialize(fd, corpus)	
	
	""" Guardar el diccionario en memoria """
	def _save_dictionary(fd): self.__dictionary.save(fd)
	
	"""Retornar la representación bag-of-words (vector-disperso de pares (id,nºocurrencias) por documento) empleando 
	   one-document-at-time - memory-friendly - """
	def __iter__(self):
		for document in listdir(self.__test_corpus_path):
			fd = open(self.__test_corpus_path+"/"+document)
			yield self.__dictionary.doc2bow(tokenize(fd.read()))
			
