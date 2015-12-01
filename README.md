# HackYrNews


###About
HackYrNews is an app for building a custom news feed for Hacker News. Recommendations for posts are built around a word2vec model trained on the comment text; a post is suggested if its title is "near" the user's interests, which can be a list of words, or sums and differences of words. 


###Files
1) scrape.py is used to access the Hacker News API and get recent items (post titles, comments, and possibly other kinds of items).

2) word2vec_data.py contains an object for extracting the scraped data from the API (stored as a pickled list of python dicts), reformatting it as needed for fitting models, and fitting the models. 

3) stability.py contains tools for comparing assessing the stability of the fitted word2vec model with respect to initial condition and number of training epochs. 

4) the app folder contains all files for the app, including the front end and some helper functions needed for sorting recommendations.  
