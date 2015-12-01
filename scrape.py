import time
import requests
import json
import pickle
import datetime

'''
Original max_id was max_id = 10194295
'''


def get_recent_items(start_id, 
    max_hours=1,
    increasing=True):
    '''
    Scrape items from HN. Start with a given post id and go scrape in 
    increasing or decreasing order until max_hours hours have passed. 
    '''    
    initial_time = time.time()
    items = []
    error_ids = []
    current_id = start_id 
    i = 1
    
    while True:
        try:
            request_url = 'https://hacker-news.firebaseio.com/v0/item/' + \
                          str(current_id) + \
                          '.json?print=pretty'                     
            current_request = requests.get(request_url, 
                auth=('user', 'pass'))
            items.append(current_request.json())
        except:
            error_ids.append(current_id)
        if i % 50 == 0:
            print 'Time Elapsed: ', time.time() - initial_time
            print 'current_id: ', current_id, 'item number: ', i         
            if (time.time() - initial_time) / 3600.0 >= max_hours:
                return items, error_ids
        if increasing:
            current_id += 1  
        else: 
            current_id -= 1
        i += 1 
    
    return items, error_ids


MAX_ID = 10194295
hn_test, errors_test = get_recent_items(MAX_ID, 
                                        max_hours=5.5,
                                        increasing=False)

f = open("HN_10194295_55.p", "wb")
pickle.dump(hn_test, f)
f.close()