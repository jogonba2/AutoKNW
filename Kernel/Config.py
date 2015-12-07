#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.stem   import WordNetLemmatizer
from scipy.spatial import distance
from FeatureExtractor  import get_sentence_representation_sum, get_sentence_representation_mult,get_sentence_representation_distance,get_sentence_representation_weighted,get_sentence_representation_covariance
from sklearn import svm
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from Classifier import get_class
from Classifier import get_k_classes

## Project info ##

AUTHOR   = "JGonzalez"
VERSION  = "a0.01"

################

## General ##

VERBOSE = False

#############

## Preproces ##

LANGUAGES = ["spanish","english"]
LANGUAGE = "spanish"
STOPWORDS = []
for lang in LANGUAGES: 
	for word in stopwords.words(lang):
		STOPWORDS.append(word)
STEMMER   = SnowballStemmer(LANGUAGE)
LEMMATIZER = WordNetLemmatizer()
PATTERN_TOKENIZER = " "
PATTERN_SEP_SENTENCE  = "\n"
NON_ALFANUMERIC_PATTERNS = ["\n",".",","]
NON_ALFANUMERIC_FUNC     = 2

###############

## Word2vec model ##

MIN_COUNT_W2V = 2
WINDOW_W2V    = 1
SIZE_W2V      = 250

####################

## Feature extraction ##

SENTENCE_REPRESENTATION = get_sentence_representation_sum
DISTANCE_REPRESENTATION = distance.correlation
REGULARIZATION          = False

########################

## Classification ##

DISTANCE_CLASSIFICATION = distance.correlation
K_PROJECT_CLASSIFIER    = 1
## Classifier configs ##
# Add here #
# Example: SVM_KERNEL=radial, ...  #

# Config here #
CLASSIFIERS = {"SVM_LINEAR":svm.LinearSVC(),"NEAREST_CENTROID":NearestCentroid(metric=DISTANCE_CLASSIFICATION),"NAIVE_BAYES":GaussianNB(),"DECISSION_TREE":tree.DecisionTreeClassifier(),"PROJECT_DISTANCE_1":get_class,"PROJECT_DISTANCE_2":get_k_classes}

CLASSIFIER  = CLASSIFIERS["NEAREST_CENTROID"]
####################
