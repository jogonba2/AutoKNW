#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import walk
try: from cPickle import dump,load,HIGHEST_PROTOCOL
except: from pickle import dump,load,HIGHEST_PROTOCOL
import Config

def serialize(file_name,object): 
	with open(file_name,'wb') as fd: dump(object,fd,HIGHEST_PROTOCOL)
	
def unserialize(file_name):
	res = []
	with open(file_name,'rb') as fd: res.append(load(fd))
	return res

def get_sentences(path):
	res = []
	file_names = get_files(path)
	for filename in file_names:
		with open(filename) as fd_file:		
			for line in fd_file.readlines():
				line = line[:-1] 	
				if len(line)>1: 
					res.append((line,filename[filename.rfind("/")+1:len(filename)-4]) )
		fd_file.close()
	return res

def get_files(path):
	res = []
	for root,dirs,files in walk(path):
		for fd in files: res.append(root+"/"+fd)
	return res

def regularization(sentence,val_max_sentences,val_min_sentences): 
	return (sentence-val_min_sentences)/val_max_sentences
