import pandas as pd
import numpy as np
import ast
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# 📂 Load datasets
# ==============================
movies = pd.read_csv('tmdb_5000_movies.xls')
credits = pd.read_csv('tmdb_5000_credits.xls')

# ==============================
# 🔗 Merge datasets
# ==============================
movies = movies.merge(credits, on='title')

# ==============================
#  Select useful columns
# ==============================
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# ==============================
# 🧹 Data Cleaning
# ==============================
movies.dropna(inplace=True)


# Convert stringified list → list
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L


movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)


# Top 3 cast
def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L


movies['cast'] = movies['cast'].apply(convert_cast)


# Fetch director
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


movies['crew'] = movies['crew'].apply(fetch_director)


# Overview → split words
movies['overview'] = movies['overview'].apply(lambda x: x.split())


# Remove spaces (e.g., "Sam Worthington" → "SamWorthington")
def collapse(L):
    return [i.replace(" ", "") for i in L]


movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)


# ==============================
# 🏷️ Create tags
# ==============================
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# Final dataframe
new_df = movies[['movie_id', 'title', 'tags']]

# Convert list → string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))


# ==============================
# 🔠 Vectorization
# ==============================
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()


# ==============================
# 🤖 Similarity calculation
# ==============================
similarity = cosine_similarity(vectors)


# ==============================
# 💾 Save files
# ==============================
pickle.dump(new_df.to_dict(), open('movie_dict.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print(" movie_dict.pkl and similarity.pkl created successfully!")