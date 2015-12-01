
import pickle
from flask import render_template, request
import pandas as pd
from pandas import DataFrame, Series
from nltk import word_tokenize
from sqlalchemy import create_engine
import numpy as np
from pandas.io import sql
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from nltk import word_tokenize
import nltk 
from app import app

engine = create_engine("mysql://root@localhost/hn_100_aws?charset=utf8", 
    echo_pool=True)
connection_raw = engine.raw_connection()

story_vec_df = sql.read_frame("SELECT * FROM story_vec", connection_raw)
comment_vec_df = sql.read_frame("SELECT * FROM comment_vec", connection_raw)
token_df = sql.read_frame("SELECT token FROM model_vec", connection_raw)
connection_raw.close()

vocab = set(list(token_df.token))
vec_cols = ['vec_' + str(i) for i in range(200)]



def query_model_vec(query_token):
    local_engine = create_engine("mysql://root@localhost/hn_100_aws?charset=utf8",
        echo_pool=True)
    local_connection = local_engine.raw_connection()
    token_df = sql.read_frame("SELECT * FROM model_vec WHERE token = \'%s\' LIMIT 1" % query_token,
        local_connection)
    local_connection.close()
    return token_df
    



def is_term_valid(term, vocab = vocab):
    word = remove_space(term)
    if word.isalpha():
        word = word.lower()
        if word in vocab:
            return {'sign': '+', 'word': word, 'is_valid': True}

    elif len(word) == 0:
        return {'sign': '+', 'word': word, 'is_valid': False}
        
    elif word[0] in {'+', '-'}:
        text_word = word[1:].lower()
        if text_word.isalpha():
            if text_word in vocab:
                return {'sign': word[0], 'word': text_word, 'is_valid': True}
            
    
    return {'sign': 'NA', 'word': term, 'is_valid': False}




    
def remove_space(word):
    return "".join(word.split())

def parse_compound_term(term, vocab = vocab):
    '''
    For parsing +___|-___ terms in user input.
    '''
    if '|' in term:
        indiv_kws = term.split('|')
        neg_words = [word for word in indiv_kws if word[0] == '-']
        pos_words = [word for word in indiv_kws if word not in neg_words]

        cleaned_pos = [word_tokenize(word[1:].lower())[0] for word in pos_words
                        if word[1:].lower().isalpha()
                            and word[1:].lower() in vocab]

        cleaned_neg = [word_tokenize(word[1:].lower())[0] for word in neg_words
                        if word[1:].lower().isalpha()
                            and word[1:].lower() in vocab]

        return {'pos': cleaned_pos, 'neg' : cleaned_neg}

    else:
        return word_tokenize(term.lower())


def input_tokenizer(input_string):
    '''
    Tokenize input, determine which terms are valid. 
    '''
    possible_terms = input_string.split(',') 
    kw_output = []
    bad_terms = []
    for item in possible_terms:
        subterms = item.split('|')    
        term_dict = {'pos': [], 'neg': []}
        for token in subterms:
            if is_term_valid(token)['is_valid'] == False and len(token) > 0:
                bad_terms.append(token.strip())
            else:
                if is_term_valid(token)['sign'] == '+' and len(token) > 0:
                    term_dict['pos'].append(is_term_valid(token)['word'])
                elif is_term_valid(token)['sign'] == '-' and len(token) > 0:    
                    term_dict['neg'].append(is_term_valid(token)['word'])
                    
        kw_output.append(term_dict)
    return kw_output, bad_terms                
                    
 
def text_kw_from_string(raw_kw):
    if type(raw_kw) == list:
        return raw_kw[0]
        
    elif type(raw_kw) == dict and set(raw_kw.keys()) == {'pos', 'neg'}:
        pos_text = '+' + '|+'.join(raw_kw['pos'])
        neg_text =  '-' + '|-'.join(raw_kw['neg']) 

        if len(raw_kw['pos']) >= 1 and len(raw_kw['neg']) >= 1:
            return pos_text + '|' + neg_text
        
        elif len(raw_kw['pos']) >= 1 and len(raw_kw['neg'])  ==  0:
            return pos_text 
            
        elif len(raw_kw['pos']) == 0 and len(raw_kw['neg']) >= 1:
            return neg_text
            

def pretty_print_input(raw_string):
    '''
    For display of input on landing page.
    '''
    good_kw_dicts =  input_tokenizer(raw_string)[0]
    error_kws =  input_tokenizer(raw_string)[1]
    good_kws_pretty = [text_kw_from_string(kw_dict) for kw_dict in good_kw_dicts
                       if text_kw_from_string(kw_dict) != None]

    error_kws = [kw for kw in error_kws if len(kw.strip()) > 0]                
    return {'good_kws': good_kws_pretty, 'error_kws': error_kws}




def get_recs(keyword,
    story_vector_df,
    model_query_fn = query_model_vec,
    reweighting = 'naive'):
    '''
    Get recommendations for a single keywoed
    '''
    if type(keyword) == list:

        kw_dfs = [model_query_fn(kw).ix[:, vec_cols] for kw in keyword]
        kw_df = pd.concat(kw_dfs, ignore_index = True)

        kw_vec = kw_df.apply(np.mean, 0)
        similarity = story_vector_df.ix[:, vec_cols].apply(lambda x: np.dot(x, kw_vec), 1)
        temp_cols = ['story_id', 
                     'story_title', 
                     'similarity', 
                     'url', 
                     'time', 
                     'readable_t']        
        temp = story_vector_df.ix[:,temp_cols]
        temp['similarity'] = similarity
        temp['tag'] = text_kw_from_string(keyword)
        temp = temp.dropna()
        return temp.sort(columns = ['similarity'], ascending = False)
    

    elif type(keyword) == dict and set(keyword.keys()) == {'pos', 'neg'}:
        pos_kws = keyword['pos']
        neg_kws = keyword['neg']
        pos_neg = []
        if len(pos_kws) > 0:
            pos_dfs = [model_query_fn(kw).ix[:, vec_cols] for kw in pos_kws]
            pos_df = len(pos_kws) * pd.concat(pos_dfs, ignore_index = True)
            pos_neg = pos_neg + [pos_df]
                
        if len(neg_kws) > 0:        
            neg_dfs = [model_query_fn(kw).ix[:, vec_cols] for kw in neg_kws]
            neg_df = -len(neg_kws) * pd.concat(neg_dfs, ignore_index = True)
            pos_neg = pos_neg + [neg_df]
            
        if len(neg_kws) == 0 and len(pos_kws) == 0:
            return None

        full_df = pd.concat(pos_neg, ignore_index = True)
        kw_vec = full_df.apply(np.mean, 0)

        similarity = story_vector_df.ix[:, vec_cols].apply(lambda x: np.dot(x, kw_vec), 1)
        temp = story_vector_df.ix[:,['story_id', 
                                     'story_title', 
                                     'similarity', 
                                     'url', 
                                     'time', 
                                     'readable_t',
                                      'num_comments']]
        temp['similarity'] = similarity
        temp['tag'] = text_kw_from_string(keyword)
        temp = temp.dropna()
        return temp.sort(columns = ['similarity'], ascending = False)


def comment_url_from_id(story_id):
    return 'https://news.ycombinator.com/item?id=' + str(story_id)
            
            
    
def get_merged_recommendations(keywords,
    get_recs_from_kw,
    total_results = 30,
    similarity_threshold = 0.7,
    normalizer = np.sum):
    '''
    Merge recommendations coming from different search terms. 
    '''
    rec_dfs = []

    for keyword in keywords:
        if len(keyword['pos']) or len(keyword['neg']):
        
        
            df = get_recs_from_kw(keyword).sort('similarity', ascending = False)
            filtered = df.head(total_results)
            filtered_thresh = filtered.ix[filtered.similarity >- similarity_threshold, :]
            filtered.shape        
            filtered.similarity = filtered.similarity/normalizer(filtered.similarity)
            rec_dfs.append(filtered)
    
    
    main_df = pd.concat(rec_dfs, ignore_index = True)
    main_df['comment_url'] = main_df.story_id.apply(comment_url_from_id)
    return main_df.sort(columns = ['similarity'], ascending = False)




def get_recs_local(token_input):
    return get_recs(token_input, story_vec_df)


def get_merged_recs_local(keywords, get_recs_local):
    return get_merged_recommendations(keywords, get_recs_local)

    
def hn_ranking(story_data, 
    gravity = 1.8,
    max_time = 1441690426):

    hours_elapsed = (max_time - story_data['time'])/(3600.0)
    score = story_data['score']
    return (story_score - 1)/(hours_elapsed)**gravity


@app.route('/')
def cities_input():  
    return render_template("index.html")

@app.route('/output')   
def cities_output():
    
    input_text = request.args.get('ID')
    #Note: at this point we treat all kws separately and then merge. 
    input_words = input_tokenizer(input_text)[0]
    recs = get_merged_recommendations(input_words, 
        get_recs_local)
    bad_words = input_text[1]
    kw_dicts = input_words[0]
    prettified_input = pretty_print_input(input_text)
    good_kws = prettified_input['good_kws']
    error_kws = prettified_input['error_kws']
    the_result = [text_kw_from_string(kw_dict) for kw_dict in kw_dicts
                    if text_kw_from_string(kw_dict) != None]    
    invalid_kws_exist = len(error_kws) > 0
    
    rec_rows = []
    for i, row in recs.iterrows():
        rec_rows.append(dict(story_title = row['story_title'], 
                             similarity =row['similarity'],
                             url = row['url'],
			     story_id = row['story_id'], 
                             time = row['readable_t'],
                             tag = row['tag'],
                             num_comments = row['num_comments'],
                             comment_url = row['comment_url']))
    return render_template("output.html", 
                            rec_rows = rec_rows, 
                            bad_words = error_kws, 
                            the_result = good_kws,
                            invalid_kws_exist = invalid_kws_exist)
  
