import numpy as np
import math
import random
import pickle
import datetime
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from gensim import corpora, models, matutils 
from nltk import word_tokenize 
from gensim.models import doc2vec
from gensim import models
from nltk.corpus import stopwords

class Word2VecData:
    
    '''
    class for extracting data from the json dics, formatting/cleaning the
    results for Doc2Vec, writing results to csv, and fitting word2vec models
    on the data.     
    '''        
    def __init__(self, scrape_file, min_length = 3):
        
        f = open(scrape_file, 'rb')
        recent_items = pickle.load(f)
        f.close()
    
        item_dic = {item['id']: item for item in recent_items
                    if item and 'id' in item.keys()}

        def get_items_of_a_type(item_list, item_type):
            result = [item for item in item_list
                          if item
                              and item['type']
                              and item['type'] == item_type]
    
            return result

        recent_stories = get_items_of_a_type(recent_items, 'story')
        recent_comments = get_items_of_a_type(recent_items, 'comment')

        
        
        
        
        comment_text = [(comment['text'], comment['id']) 
                        for comment in recent_comments
                        if comment 
                            and 'id' in comment.keys()
                            and 'text' in comment.keys()]


        comment_df = DataFrame({
                                'id' :  [item[1] for item in comment_text],
                                'raw_text' : [item[0] for item in comment_text]
                               })
        '''
        Each comment should really belong to a parent story
        '''


        def get_ancestor_data(comment_id, field = 'id'):
            current_item = item_dic[comment_id]
            while current_item['type'] and current_item['type'] == 'comment':
                try:
                    parent_id = current_item['parent']
                    current_item = item_dic[parent_id]
                except:
                    print 'Problem finding parent, item id: ', current_item['id']
                    return 'NaN'
        
            try:
                return current_item[field]
            except:
                print 'Problem finding field in ancestor ', current_item['id']
                
          
        comment_df['parent_id'] = comment_df['id'].apply(lambda x: get_ancestor_data(x)) 
        comment_df['parent_title'] = comment_df['id'].apply(lambda x: get_ancestor_data(x, field = 'title')) 
        comment_df['parent_type'] = comment_df['id'].apply(lambda x: get_ancestor_data(x, field = 'type')) 
        comment_df = comment_df[comment_df.parent_type == 'story']
        comment_df = comment_df[~pd.isnull(comment_df.parent_title)]


        def get_property(story_id, field='url'):
            try:
                return item_dic[story_id][field]        
            except:
                return 'NaN'
                
        
        story_ids = [story['id'] 
                        for story in recent_stories
                        if story 
                            and 'id' in story.keys()
                            and 'text' in story.keys()]
                                                        
    
        story_df = DataFrame({'story_id': story_ids})  
              
        story_df['story_title'] = story_df.story_id.apply(lambda x: item_dic[x]['title'])
        story_df['num_comments'] = story_df.story_id.apply(lambda x: comment_df.ix[comment_df.parent_id == x, :].shape[0])
        story_df['url'] = story_df.story_id.apply(lambda x: get_property(x))
        story_df['time'] = story_df.story_id.apply(lambda x: get_property(x, field= 'time'))
        story_df['score'] = story_df.story_id.apply(lambda x: get_property(x, field = 'score'))    

        def get_readable_time(timestamp):
            dt = datetime.datetime.fromtimestamp(timestamp)
            dt_string = dt.strftime('%H:%M, %m-%d-%Y')
            return dt_string


        story_df['readable_t'] = story_df.time.apply(get_readable_time)
        
        
        
        
        def sentence_tokenizer(s, banned_words = {'http', 'https'}):
            return [word for word in word_tokenize(s.lower()) 
                        if word.isalpha() 
                            and word not in banned_words
                            and len(word) >= min_length]
            
        comment_df['token_text'] = comment_df.raw_text.apply(sentence_tokenizer) 
        story_df['token_title'] = story_df.story_title.apply(sentence_tokenizer)
        
        self.stories = []
        self.documents = []
        

        for i, row in comment_df.iterrows():
            new_sentence = doc2vec.LabeledSentence(words = row['token_text'],
                                                   tags = [str(row['parent_id'])])
            self.documents.append(new_sentence)                                          
        
        for i, row in story_df.iterrows():
            new_sentence = doc2vec.LabeledSentence(words = row['token_title'],
                                                   tags = [str(row['story_id'])])
            self.stories.append(new_sentence)                                          
        
        self.comment_df = comment_df
        self.story_df = story_df
        
      
                    
    def fit_w2v(self,
        min_count=20, 
        window=10, 
        size=200, 
        sample=1e-4, 
        negative=5, 
        workers=7, 
        seed=1,
        random_subset_prop = False):
        '''
        Fit word2vec model on the given data. 
        '''
        sentences = list(self.comment_df.token_text)
        
        model = models.Word2Vec(min_count = min_count, 
                window=window, 
                size=size, 
                sample=sample, 
                negative=negative, 
                workers=workers,
                seed=seed)
        model.build_vocab(sentences)
        self.model_w2v_size = size
        model.train(sentences)
        self.model = model
        
        
    def data_to_csv(self,
        comment_filename,
        story_filename):
        '''
        Write the story and comment data to text for future use.
        '''
        self.comment_df.to_csv(comment_filename + 'tsv', 
            encoding='utf-8', 
            sep='\t')
        self.story_df.to_csv(story_filename + 'tsv', 
            encoding='utf-8', 
            sep='\t')
            

    