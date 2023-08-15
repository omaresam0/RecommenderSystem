import pandas as pd
from sklearn.cluster import KMeans

#---PreProcessing---

moviesPath = pd.read_csv('movies.csv')
ratingsPath = pd.read_csv('ratings.csv')
#print(ratingsPath.shape)
moviesPath.dropna(axis=1)
ratingsPath.dropna(axis=1)
#print(ratingsPath.shape)

# no need to use one hot encoding
# becauce no much variance comparing to attributes like in tags df
# so we will apply label encoding
GenreList = {'Action', 'Adventure', 'Animation',
                 'Children', 'Comedy', 'Crime',
                 'Documentary', 'Drama', 'Fantasy',
                 'Film-Noir', 'Mystery', 'Horror',
                 'Musical', 'Romance', 'Sci-Fi',
                 'Thriller', 'War', 'Western',
                 'IMAX', '(no genres listed)'
                }

# adding 0 values for all genres
for genres in GenreList:
    moviesPath[genres] = 0

# splitting genres
# if the movie fulfill some genre, then add its value with 1
for index, row in moviesPath.iterrows():
    movieGenre = row['genres'].split('|')
    for genres in movieGenre:
        moviesPath.at[index, genres] = 1

#removing 'genres' columns
del moviesPath['genres']

#---------------



# calc average raings for each movie and append them in moviesPath as a feature
# find mean of rating of that movie (movieId)
def AvgRating(movieID):
    return ratingsPath.loc[ratingsPath['movieId'] == movieID]['rating'].mean()

AvgRating = moviesPath['movieId'].map(lambda x: AvgRating(x))
moviesPath['rating'] = AvgRating

dataGenres = moviesPath.drop(['movieId','title', 'rating'], axis =1)
dataGenres.dropna(inplace = True)
#print(dataGenres.columns)
#print(moviesPath.columns)


#20 genres, so 20 clusters or more
n_clusters = 25
kmeans = KMeans(n_clusters=n_clusters)
ClusterNo = kmeans.fit_predict(dataGenres)
moviesPath['ClusterNo'] = pd.DataFrame(ClusterNo)
#print(moviesPath.columns)

def GetRrecommendations(movieTitle):
    ClusterNum = moviesPath[moviesPath['title'] == movieTitle]['ClusterNo'].values[0] #cluster number of the recommended movies
    movies = moviesPath[moviesPath['ClusterNo'] == ClusterNum] #movies to be reocmmended
    Recommendations = movies.sort_values(by='rating', ascending=False).head(10) # top 10 movies

    #dropping all columns(genres with 0s and 1s) except for only three.
    Recommendations = Recommendations.filter(['title', 'rating', 'ClusterNo'])

    #print(Recommendations.columns)
    return Recommendations

# movie_name = input("Enter a movie name: ")
# print(GetRrecommendations(movie_name))
print(GetRrecommendations('Toy Story (1995)'))