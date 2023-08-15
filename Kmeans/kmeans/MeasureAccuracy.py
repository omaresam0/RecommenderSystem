import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
import random
# ---PreProcessing---
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from time import perf_counter
start1 = perf_counter()

moviesPath = pd.read_csv('movies.csv')
ratingsPath = pd.read_csv('ratings.csv')
# print(ratingsPath.shape)
moviesPath.dropna(axis=1)
ratingsPath.dropna(axis=1)

GenreList = {'Action', 'Adventure', 'Animation',
                 'Children', 'Comedy', 'Crime',
                 'Documentary', 'Drama', 'Fantasy',
                 'Film-Noir', 'Mystery', 'Horror',
                 'Musical', 'Romance', 'Sci-Fi',
                 'Thriller', 'War', 'Western',
                 'IMAX', '(no genres listed)'
                 }

# splitting genres
for genres in GenreList:
    moviesPath[genres] = 0

# splitting genres
# if the movie fulfill some genre, then add its value with 1
for index, row in moviesPath.iterrows():
    movieGenre = row['genres'].split('|')
    for genres in movieGenre:
        moviesPath.at[index, genres] = 1

# removing genres columns
del moviesPath['genres']

#---------------

# calc average raings for each movie and append them in moviesPath as a feature
# find mean of rating of that movie (movieId)
def AvgRating(movieID):
    return ratingsPath.loc[ratingsPath['movieId'] == movieID]['rating'].mean()

AvgRatinging = moviesPath['movieId'].map(lambda x: AvgRating(x))
moviesPath['rating'] = AvgRatinging

dataGenres = moviesPath.drop(['movieId', 'title', 'rating'], axis=1)
dataGenres.dropna(inplace=True)
#print(dataGenres.columns)
#print(moviesPath.columns)

wcss = []  # WCSS Values' List

# from 1 to 30
max = 30
for i in range(1,max+1):
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(dataGenres)
    wcss.append(km.inertia_)




# 20 genres, so 20 clusters or more
n_clusters = 25
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
ClusterNo = kmeans.fit_predict(dataGenres)
moviesPath['ClusterNo'] = pd.DataFrame(ClusterNo)
dataGenres['labels'] = ClusterNo


X = dataGenres.iloc[:, :-1]
y = dataGenres.iloc[:, -1]
#print(X.columns)

# 75% trains-set / 25% test-set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# to get same results every time
np.random.seed(0)
random.seed(0)

ss = StandardScaler(with_mean=False)
ss.fit(X_train)
X_train = ss.transform(X_train)
X_test = ss.transform(X_test)

# fit the model
lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred= lr.predict(X_test)

# round value after 2 decimal
print("Accuracy:", accuracy_score(y_pred, y_test))

clustersNum = ["Cluster {}".format(i) for i in range(n_clusters)]
print(classification_report(y_test, y_pred, target_names=clustersNum))

# Accuracy Metrics
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
MAE = np.sqrt(mean_absolute_error(y_test, y_pred))

print("RMSE:", RMSE)
print("MAE:", MAE)

end = perf_counter()
print(f"exection time:{end - start1}")

# Elbow Method
plt.plot(range(1,max+1), wcss, 'go-')
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of Squared Error (SSE)')
plt.show()

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), fmt='', annot=True)
plt.show()
