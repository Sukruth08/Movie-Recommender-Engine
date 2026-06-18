import streamlit as st
import pandas as pd
import pickle
import requests
from concurrent.futures import ThreadPoolExecutor

# ==============================
# 🔑 TMDB API Key
# ==============================
API_KEY = "b7d3955b180a1d71a2dd6949cd41140a"

# ==============================
# 🎬 Fetch movie poster (FINAL FIX)
# ==============================
@st.cache_data(show_spinner=False)
def fetch_poster(movie_name):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_name}"
        response = requests.get(url, timeout=5)

        data = response.json()

        if data and data.get('results'):
            poster_path = data['results'][0].get('poster_path')

            if poster_path:
                return "https://image.tmdb.org/t/p/w500" + poster_path

    except Exception as e:
        print("Error:", e)

    return "https://via.placeholder.com/500x750?text=No+Poster"


# ==============================
# 📂 Load data
# ==============================
@st.cache_data
def load_data():
    try:
        movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
        similarity = pickle.load(open('similarity.pkl', 'rb'))

        movies = pd.DataFrame(movies_dict)
        movies['title'] = movies['title'].str.strip()

        return movies, similarity

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None


movies, similarity = load_data()

if movies is None or similarity is None:
    st.stop()


# ==============================
# 🤖 Recommendation function
# ==============================
def recommend(movie):
    try:
        movie = movie.strip()

        if movie not in movies['title'].values:
            return [], []

        movie_index = movies[movies['title'] == movie].index[0]
        distances = similarity[movie_index]

        movies_list = sorted(
            list(enumerate(distances)),
            key=lambda x: x[1],
            reverse=True
        )[1:6]

        recommended_movies = []
        movie_names = []

        for i in movies_list:
            recommended_movies.append(movies.iloc[i[0]]['title'])
            movie_names.append(movies.iloc[i[0]]['title'])

        # ⚡ Parallel fetching
        with ThreadPoolExecutor(max_workers=5) as executor:
            recommended_posters = list(executor.map(fetch_poster, movie_names))

        return recommended_movies, recommended_posters

    except Exception as e:
        st.error(f"Error in recommendation: {e}")
        return [], []


# ==============================
# 🎨 Streamlit UI
# ==============================
st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommender System")

selected_movie = st.selectbox(
    "Choose a movie",
    movies['title'].values
)

if st.button("Recommend"):
    with st.spinner("Finding recommendations..."):
        names, posters = recommend(selected_movie)

    if len(names) > 0:
        cols = st.columns(5)

        for i in range(len(names)):
            with cols[i]:
                st.markdown(f"**{names[i]}**")
                st.image(posters[i])  # ✅ fixed

    else:
        st.warning("No recommendations found!")