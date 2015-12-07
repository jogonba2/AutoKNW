#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from rdflib import Graph,URIRef,Namespace
from scipy.spatial import distance
## Hacer funciones para parsear frases de la forma correcta y poder consultar en /resource y generar el grafo sobre las frases del corpus ##
## Se requiere preproceso de las palabras entre mayusculas etc. "
g                  = Graph()
resources          = ["http://dbpedia.org/resource/"+word for word in ["Mariano_Rajoy","Rafa_Nadal","Barack_Obama","Angela_Merkel","Madrid","Valencia","Italia","FC_Bayern_Munich","Francia","Liverpool_F.C."]]
features           = set()
res_features       = {}

# Con esta linea se lee directamente el grafo (cuando ya se haya cargado desde la dbpedia) #
#graphs = g.parse("graphs.txt",format='turtle')

## Cargar grafo inicial y almacenarlo si no se tiene! ##
for resource in resources: graphs = g.parse(resource) # Los grafos se van combinando #
graphs.serialize(destination='graphs.txt', format='turtle')

# Cargar caracteristicas #
for subj,pred,obj in graphs: features.add(str(pred))
print features
# Obtener representacion vectorial #	
struct_query = "select ?res WHERE{ _SUBJECT_ _PREDICATE_ ?res . }"
for resource in resources:
	res_features[resource] = []
	query = struct_query.replace("_SUBJECT_","<"+resource+">")
	for feature in features:
		query2 = query.replace("_PREDICATE_","<"+feature+">")
		if len(graphs.query(query2))>=1: res_features[resource].append(1)
		else:						     res_features[resource].append(0)

def get_most_similar(v1,res_features):
	dist,best = float("inf"),None
	for i in res_features:
		if v1!=res_features[i]:
			distancia = distance.correlation(v1,res_features[i])
			if distancia<dist: dist,best = distancia,i
	return best

for i in res_features:
	print "MAS SIMILAR DE",i,"->",get_most_similar(res_features[i],res_features)
