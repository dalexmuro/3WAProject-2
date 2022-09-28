import json
import pickle
import requests
import numpy as np
import pandas as pd

from os.path import exists
from cleantext import cleantext

# API url's
url_reviews_api = 'https://imdb-api.com/en/API/Reviews/k_af2q00ae/'
url_rating_api = 'https://imdb-api.com/en/API/Ratings/k_af2q00ae/'

# Local paths
full_data_path = 'data/full_data.plk'
csv_path = 'data/dataset/mpst_full_data.csv'
model_sentiment_path = 'data/model_sentiment.sav'
# tfIdfOverview_path = 'data/overviews.plk'
# tfIdfScores_path = 'data/scores.plk'
# sig_path = 'data/sig.plk'

# Pipeline with the model and tokenizer created with sent_analysis.ipynb
model_sent = pickle.load(open(model_sentiment_path, 'rb'))

def getData():
    
    if exists(full_data_path):
        full_movies_df = pd.read_pickle(full_data_path)
        print(f'File {full_data_path} opened successfully')
    
    else:
        movies_df = pd.read_csv(csv_path)
        full_movies_df = movies_df.drop(
            columns=['tags', 'split', 'synopsis_source']
        )
        full_movies_df['reviews'] = np.empty((len(full_movies_df), 0)).tolist()
        full_movies_df['imDb_rate'] = 0
        full_movies_df['metacritic_rate'] = 0
        full_movies_df['theMovieDb_rate'] = 0
        full_movies_df['rottenTomatoes_rate'] = 0
        full_movies_df['filmAffinity_rate'] = 0
        full_movies_df['reviews_avg_rate'] = 0
        full_movies_df['sentiment_avg_rate'] = 0
        
        for i in range(len(full_movies_df)):
            # If the film hasn't been treated yet
            if(len(full_movies_df.loc[i, 'reviews']) == 0):
                
                # get the id
                movie_id = full_movies_df.loc[i, 'imdb_id']
                
                # Build the paths for the API requests
                url_review = url_reviews_api + movie_id
                url_rating = url_rating_api + movie_id
                
                # Get the data for that movie
                reviews = requests.get(url_review).text
                rating = requests.get(url_rating).text
                
                json_reviews = json.loads(reviews)
                json_rating = json.loads(rating)
                
                # Check if the ratings where properly gotten from the API and, if so, 
                # then insert the rating values into the dataframe using a 0-10 scale, or zero si no value is defined, 
                if(json_rating['title'] != None):
                
                    full_movies_df.loc[i, 'imDb_rate'] = float(json_rating['imDb'] or 0)
                    full_movies_df.loc[i, 'metacritic_rate'] = float(json_rating['metacritic'] or 0)/10 # The original scale is based over 100 instead of 10
                    full_movies_df.loc[i, 'theMovieDb_rate'] = float(json_rating['theMovieDb'] or 0)
                    full_movies_df.loc[i, 'rottenTomatoes_rate'] = float(json_rating['rottenTomatoes'] or 0)/10 # The original scale is based over 100 instead of 10
                    full_movies_df.loc[i, 'filmAffinity_rate'] = float(json_rating['filmAffinity'] or 0)
                
                # Check if there are reviews gotten from the API
                if(json_reviews['title'] != None and len(json_reviews['items']) > 0):
                    rate = 0
                    rates_count = 0
                    sentiment_total = 0

                    # Calculate the average rating for all the reviews
                    for review in json_reviews['items']:
                        # Frist, add each review to the list
                        full_movies_df.loc[i, 'reviews'].append(review['content'])
                        
                        # Clean the text of the review
                        cleaned_review = [cleantext(review['content'])]
                        
                        #Get the sentiment usin the model
                        sentiment = model_sent.predict(cleaned_review.toarray())
                        
                        # Ad the sentiment to the total
                        sentiment_total += int(sentiment)*10 # Covert 1/0 into 10/0 in order to be in the same scale

                        # If there is a rate of that review, then add it to the total and increase the count
                        if (review['rate'] != ''):
                            rate += int(review['rate'])
                            rates_count += 1
                    
                    # Get the averages
                    full_movies_df.loc[i, 'reviews_avg_rate'] = round(rate/rates_count, 1) if rates_count > 0 else 0
                    full_movies_df.loc[i, 'sentiment_avg_rate'] = round(sentiment_total/len(json_reviews['items']), 1)
                    
                    # Set an array with all the ratings and then get the average
                    all_rates = full_movies_df.loc[i].filter(items=['imDb_rate',
                                                                    'metacritic_rate',
                                                                        'theMovieDb_rate',
                                                                        'rottenTomatoes_rate',
                                                                        'filmAffinity_rate',
                                                                        'reviews_avg_rate',
                                                                        'sentiment_avg_rate']
                                                                ).to_numpy()
                    
                    full_movies_df.loc[i, 'avg_rate'] = round(all_rates[np.nonzero(all_rates)].mean(), 2)
                    
                    print('Rates saved for the film : ' + full_movies_df.loc[i, 'title'])
                
                else:
                    print('No reviews or rating found for the film : ' + full_movies_df.loc[i, 'title'])
                    full_movies_df = full_movies_df.drop(i).reset_index(drop=True)
                
        # Save the Dataframe
        full_movies_df.to_pickle(full_data_path)
        print(f'File {full_data_path} created successfully')
    
    return full_movies_df