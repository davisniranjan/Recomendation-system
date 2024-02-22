import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import warnings


# Ignore future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load data
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

# Create user-item matrix
user_mapper = {user_id: index for index, user_id in enumerate(ratings['userId'].unique())}
movie_mapper = {movie_id: index for index, movie_id in enumerate(ratings['movieId'].unique())}
user_index = ratings['userId'].map(user_mapper)
movie_index = ratings['movieId'].map(movie_mapper)
X = csr_matrix((ratings['rating'], (movie_index, user_index)), shape=(len(movie_mapper), len(user_mapper)))

# Find similar movies using KNN
kNN = NearestNeighbors(n_neighbors=11, algorithm="brute", metric='cosine')
kNN.fit(X)

def find_similar_movies(movie_id):
    movie_ind = movie_mapper.get(movie_id)
    if movie_ind is None:
        return []
    movie_vec = X[movie_ind]
    _, neighbour_ind = kNN.kneighbors(movie_vec)
    neighbour_ids = [list(movie_mapper.keys())[i] for i in neighbour_ind.flatten() if i != movie_ind]
    return neighbour_ids
def on_recommend():
  user_id = entry_user_id.get()
  if user_id.isdigit():
    recommend_movies_for_user(int(user_id))
  else:
    messagebox.showerror("Error", "Please enter a valid user ID.")

def recommend_movies_for_user(user_id):
  user_ratings = ratings[ratings['userId'] == user_id]
  if user_ratings.empty:
      messagebox.showerror("Error", f"User with ID {user_id} does not exist.")
      return
  max_rated_movie_id = user_ratings.loc[user_ratings['rating'].idxmax(), 'movieId']
  similar_movies = find_similar_movies(max_rated_movie_id)
  movie_titles = dict(zip(movies['movieId'], movies['title']))
  recommended_movies = [movie_titles.get(movie_id, 'Unknown') for movie_id in similar_movies]
  recommended_movies_text = '\n'.join(recommended_movies)
  messagebox.showinfo("Recommendations", f"Recommended movies for user {user_id}:\n\n{recommended_movies_text}")


# Create GUI
root = tk.Tk()
root.title("Movie Recommendation System")
root.geometry("700x500")
root.config(bg="#a5a5a5")
root.resizable(width=False,height=False)

label_user_id = tk.Label(root, text="Enter User ID:")
label_user_id.grid(row=0, column=0, padx=12, pady=7)
 
entry_user_id = tk.Entry(root)
entry_user_id.grid(row=0, column=0, padx=12, pady=7)

button_recommend = tk.Button(root, text="Recommend Movies", command=on_recommend)
button_recommend.grid(row=1, column=0, columnspan=2, pady=10)

entry_user_id.place(x=300,y=75)
button_recommend.place(x=125,y=125)
label_user_id.place(x=170,y=75)

root.mainloop()
