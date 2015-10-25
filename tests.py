#!/usr/bin/env python
# -*- coding: utf-8 -*-

from include import *


""" Generar diccionario y corpus (representacion bag of words) """
corpus_memory_friendly = CorpusWrapper(train_corpus_path="./Corpus/Corpus_1/",test_corpus_path="./Corpus/Corpus_1/")

""" Obtener la representacion bag of words de un corpus de test (se emplea el mismo que para entrenamiento) """
bag_of_corpus = corpus_memory_friendly._get_test_corpus_data()
print "\n\nRepresentacion BAGOFWORDS: ",bag_of_corpus

""" Obtener diccionario """
dictionary  = corpus_memory_friendly._get_dictionary()

""" Inicializar el modelo sobre el corpus de entrenamiento """
corpus_transformations = CorpusTransformation(bag_of_corpus,dictionary)

""" Configurar modelo LSI """
corpus_transformations._set_lsi_model(2)

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
print "\n\n El mas similar para cada documento es: ",corpus_transformations._more_similar_topic_for_documents(lsi_representation)

""" Entrenamiento online con el mismo corpus de entrenamiento """
new_lsi_model = corpus_transformations._add_documents_to_model(model_lsi,bag_of_corpus)
corpus_transformations._set_lsi_model(new_lsi_model)

""" Volver a comprobar la representacion de un modelo (LSI) de un corpus entero (también es posible sobre un único vector en cualquier modelo) """
lsi_representation = corpus_transformations._apply_transform_to_vector(model_lsi,bag_of_corpus)
print "\n\nRepresentacion LSI (sobre corpus) despues de entrenamiento online: ",lsi_representation
