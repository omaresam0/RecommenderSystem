import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from surprise import SVD, Reader, Dataset
from surprise.model_selection import cross_validate
import random

ratingsPath = 'ratings.csv'
moviesPath = 'movies.csv'

movies = pd.read_csv(moviesPath)
ratings = pd.read_csv(ratingsPath)

ratings.drop(['timestamp'], 1, inplace=True)

ratings2 = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(
    0)

np.random.seed(20)
random.seed(0)


U, E, V = svds(ratings2)  # Matrix Factorization

E = np.diag(E)  # build a diagonal matrix (sigma)
A_Matrix = np.dot(np.dot(U, E), V)  # Construct matrix through dot product  A= UEV
predictions = pd.DataFrame(A_Matrix, columns=ratings2.columns)  # convert it to a DF

svdModel = SVD(n_factors=0)  # use svd algorithm - to be evaluated

# Cross-Validation
dataReader = Reader(rating_scale=(1, 5))
user_ratings = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], dataReader)  # Read data
print()
cross_validate(svdModel, user_ratings, measures=['RMSE', 'MAE'], cv=5,
               verbose=True)  #5-folds
print()



# ----Getting Recommendations----

print("Enter user id")
print()
userid = int(input())  # user id
row = userid - 1  # skip header row
user_predictions = predictions.iloc[row].sort_values(ascending=False)  # retrieve rating predictions for a user

UserRatings = ratings[ratings.userId == (userid)]  # retrieve current user ratings and pass it to UserMovieRatings

# Merge Ratings with movie ids [sorted highest ratings to lowest]
UserMovieRatings = pd.merge(UserRatings, movies, how='left', left_on='movieId', right_on='movieId').sort_values(
    ['rating'], ascending=False)
UserMovieRatings2 = pd.merge(UserRatings, movies, how='left', left_on='movieId', right_on='movieId').sort_values(
    ['rating'], ascending=False) #needed to compare with movieId (which will be dropped in UserMovieRatings)

# Movies rated by the user..
print('User {0}: You have rated {1} films'.format(userid, UserMovieRatings.shape[0]))
print('------------------------------------')
UserMovieRatings.set_index('title', inplace=True) #set row to title
UserMovieRatings.drop(columns=['userId', 'movieId', 'genres'], inplace=True) #only need title and ratings
print(UserMovieRatings.head(10))  # top 10 movie rated by the user
print()


recommendations = 10

print('Top {0} recommendations for user {1}'.format(recommendations, userid))
print()

# Filter recommendations from already rated movies by the user by using the bitwise not operator, then merge with user predictions
TopRecommendations = (
    movies[~movies['movieId'].isin(UserMovieRatings2['movieId'])].merge(pd.DataFrame(user_predictions),
                                                                         how='left', left_on='movieId',
                                                                         right_on='movieId'))
TopRecommendations.set_index('title', inplace=True)  # set row to title
TopRecommendations.drop(columns=['movieId', 'genres'], inplace=True)  #only title and user_predictions columns are left
TopRecommendations.sort_values(by=[row], ascending=False,
                                     inplace=True)  # get highest rating predictions
TopRecommendations.rename(columns={row: 'Rating Predictions'}, inplace=True)

print(TopRecommendations.head(recommendations))  # retrieve rating predictions


