#!/usr/bin/env python
# -*- coding: utf-8 -*-

from include import *


""" Generar diccionario y corpus (representacion bag of words) """
corpus_memory_friendly = CorpusWrapper(train_corpus_path="./Corpus/Corpus_1/",test_corpus_path="./Corpus/Corpus_1/")

""" Obtener la representacion bag of words de un corpus de test (se emplea el mismo que para entrenamiento) """
bag_of_corpus = corpus_memory_friendly._get_test_corpus_data()
print "\n\nRepresentacion BAGOFWORDS: "
for word in bag_of_corpus:
	print word
	
""" Obtener diccionario """
dictionary  = corpus_memory_friendly._get_dictionary()

""" Inicializar el modelo sobre el corpus de entrenamiento """
corpus_transformations = CorpusTransformation(bag_of_corpus,dictionary)
model_tfidf = corpus_transformations._get_tfidf_model()

""" Configurar modelo LSI (empleando el corpus tfidf, se le puede pasar tambien la representacion bag-of-words - sin 2º param -)"""
corpus_tfidf = corpus_transformations._apply_transform_to_vector(model_tfidf,bag_of_corpus)
corpus_transformations._set_lsi_model(2,corpus_tfidf)
# corpus_transformations._set_lsi_model(2,corpus_tfidf) pasandole corpus bag of words #

""" Obtener modelo LSI """
model_lsi = corpus_transformations._get_lsi_model()
	
## Visualizar los topics del modelo LSI ## 
print "\n\nTopics:"
for topic in model_lsi.print_topics(2):
	print topic,"\n"
	
""" Comprobar la representacion de un modelo (LSI) de un corpus entero (también es posible sobre un único vector en cualquier modelo) """
lsi_representation = corpus_transformations._apply_transform_to_vector(model_lsi,bag_of_corpus)
print "\n\nRepresentacion LSI (sobre corpus): ",lsi_representation

""" Comprobar los temas más similares para un conjunto de documentos """
print "\n\n*********** EXAMPLE QUERY ***********\n\n"
corpus_transformations._set_similarity_matrix(model_lsi,bag_of_corpus) # Primero configurar la matriz de similaridad con el modelo y el corpus. #
query = raw_input()
query_of_words = corpus_memory_friendly._get_bag_of_words(query) # Obtener la representacion bag of words del documento. #
query_lsi      = corpus_transformations._apply_transform_to_vector(model_lsi,query_of_words)
print corpus_transformations._similarity_query_against_corpus(query_lsi)
print "\n\n"

""" Entrenamiento online con el mismo corpus de entrenamiento """
new_lsi_model = corpus_transformations._add_documents_to_model(model_lsi,bag_of_corpus)
corpus_transformations._set_lsi_model(new_lsi_model)

""" Volver a comprobar la representacion de un modelo (LSI) de un corpus entero (también es posible sobre un único vector en cualquier modelo) """
lsi_representation = corpus_transformations._apply_transform_to_vector(model_lsi,corpus_tfidf)
print "\n\nRepresentacion LSI (sobre corpus) despues de entrenamiento online: ",lsi_representation
