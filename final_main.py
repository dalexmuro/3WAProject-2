import pickle
import pandas as pd
from os.path import exists
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

movies_cleaned_df = pd.read_pickle('data/movies.plk')
movies = movies_cleaned_df.rename(columns={'imdb_id': 'id', 'plot_synopsis' : 'overview', 'title' : 'original_title'})

final_model_path = 'data/final_model.plk'
sig_path = 'data/sig.plk'

if exists(final_model_path):
    tfv_matrix = pickle.load(open(final_model_path, 'rb'))
    print(f'File {final_model_path} opened successfully')

else :
    tfv = TfidfVectorizer(
        min_df=3,
        max_features=None,
        strip_accents="unicode",
        analyzer="word",
        token_pattern="\w{1}",
        ngram_range=(1, 3),
        stop_words="english",
    )

    tfv_matrix = tfv.fit_transform(movies["overview"])
    pickle.dump(tfv_matrix, open(final_model_path, 'wb'))
    print(f'File {final_model_path} created successfully')

if exists(sig_path) :
    sig = pickle.load(open(sig_path, 'rb'))
    print(f'File {sig_path} opened successfully')

else :
    sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
    pickle.dump(sig, open(sig_path, 'wb'))
    print(f'File {sig_path} created successfully')

indices = pd.Series(movies.index, index=movies['original_title']).drop_duplicates()

def give_rec(title, sig=sig):
    idx = indices[title]
    sig_scores = list(enumerate(sig[idx]))
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    sig_scores = sig_scores[1:11]
    movie_indices = [i[0] for i in sig_scores]
    
    result = pd.concat([
        movies['original_title'].iloc[movie_indices],
        movies['avg_rate'].iloc[movie_indices]], axis=1)
    result = result.rename(columns={'original_title': 'Title', 'avg_rate': 'Rating'})
    result = result.set_index(['Title', 'Rating'])
    result = result.sort_values(by=['Rating'], ascending=False)
    
    return result

give_rec("Mr. Holland's Opus")

print(movies.head(10)['original_title'])