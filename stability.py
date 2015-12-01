import numpy as np
import math
import random
import pickle
import datetime
import itertools
import copy
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from gensim import corpora, models, matutils 
from nltk import word_tokenize 
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import doc2vec
from gensim import models
from nltk.corpus import stopwords
import word2vec_data



def sentence_tokenizer(s, banned_words = {'http', 'https'}):
    return [word for word in word_tokenize(s.lower()) 
                if word.isalpha() 
                    and word not in banned_words
                    and len(word) >= 3]



def model_similarity(model_list,
    word_list,
    use_distance = True,
    topn = 10):
    '''
    Compare similarity of two models for neighborhoods of important words, 
    as defined by doc frequency in post titles. 
    if use_distance, then compute pct change in distance for nearest topn
        neighbors of a word
    if not use_distance, compute Jac. similarity for topn between models. 
    '''
    if not use_distance:
        word_similarities = {word: [] for word in word_list}                 
        for word in word_list:
            for indices in itertools.combinations(range(len(model_list)), 2):
                model1 = model_list[indices[0]]
                model2 = model_list[indices[1]]
                similar1 = get_word_set(model1.most_similar(word, topn = topn))
                similar2 = get_word_set(model2.most_similar(word, topn = topn))
                similarity = len(similar1.intersection(similar2))
                word_similarities[word].append(similarity)
        return word_similarities
           
           
           
    if use_distance:
        word_distances = {word: [] for word in word_list}                 
        for word in word_list:       
            for indices in itertools.combinations(range(len(model_list)), 2):
                model1 = model_list[indices[0]]
                model2 = model_list[indices[1]]                
                top_words = [item[0] for item in model1.most_similar(word, topn = topn)]
                for top_word in top_words:
                    similarity1 = model1.similarity(word, top_word)
                    similarity2 = model2.similarity(word, top_word)
                    word_distances[word].append(abs(similarity1 - similarity2)/similarity1)
        return word_distances
                
    
                


def benchmark_words(docs,
    min_keyword_df=0.003,
    max_keyword_df=0.02):
    '''
    Get a set of words to use as benchmarks 
    Words are selected by document frequency in titles or comments. 
    w2v_data argument is a Word2VecData object. 
    '''            


    kw_vectorizer = TfidfVectorizer(tokenizer = sentence_tokenizer,
        stop_words = 'english',
        min_df=min_keyword_df,
        max_df=max_keyword_df)        
    
    kw_tfidf = kw_vectorizer.fit_transform(docs)
    benchmarks = kw_vectorizer.get_feature_names()
    return benchmarks





class Word2VecStability:
    '''
    Object for comparing models trained with different initializations,
    possibly on different subsets of the training corpus.
    '''
    def __init__(self, 
        titles,
        comments,
        num_models,
        seeds,
        random_subset_prop=False,
        min_keyword_df=0.003,
        max_keyword_df=0.02):

        self.titles = titles
        self.comments = comments        
        if random_subset_prop:
            sample_num = math.floor(len(comments) * random_subset_prop)
            self.model_comments = [random.sample(comments, sample_num)
                                       for num in range(num_models)]
        self.num_models = num_models
        self.seeds = seeds
        self.random_subset_prop = random_subset_prop
        self.models = {}
        self.benchmarks = benchmark_words(self.titles, 
            min_keyword_df,
            max_keyword_df)
        self.current_epoch = 0 
    
    
    def train_models(self,
        min_count=10, 
        window=10, 
        size=200, 
        sample=1e-4, 
        negative=5, 
        workers=7, 
        epochs=1):
        '''
        Add models if none have been added yet,
        Train models for desired number of epochs, and add to dictionary of 
        models.         
        '''
        if self.current_epoch == 0:
            for i in range(self.num_models):
                new_model = models.Word2Vec(min_count = min_count, 
                    window = window, 
                    size = size, 
                    sample = sample, 
                    negative = negative, 
                    workers = workers,
                    seed = self.seeds[i])
                    
                if self.random_subset_prop:
                    new_model.build_vocab(self.model_comments[i])
                    new_model.train(self.model_comments[i])                                    
                else:
                    new_model.build_vocab(self.comments)
                    new_model.train(self.comments)                    
                self.models[(i,1)] = new_model
        
        else:
            for i in range(self.num_models):
                new_model = copy.deepcopy(self.models[(i, self.current_epoch)])
                if self.random_subset_prop:                
                    new_model.train(self.model_comments[i])                                    
                else:                
                    new_model.train(self.comments)                                    
                self.models[(i,self.current_epoch + 1)] = new_model

        self.current_epoch += 1
        
    def get_similarity(self,
        topn = 10):
        '''
        pairwise change in similarity from model to model
        '''
        return {(k1, k2): model_similarity([self.models[k1], self.models[k2]],
                        self.benchmarks,
                        topn = topn)
                for (k1, k2) in itertools.permutations(self.models.keys(), 2)}                
    

