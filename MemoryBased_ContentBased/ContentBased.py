
from surprise import AlgoBase
from surprise import PredictionImpossible
from Dataset import MoviesData
import math
import numpy as np
import heapq

class ContentBased(AlgoBase):

    #compare top 40 movies with the movie we want to perdict its rating
    def __init__(self, k=40):
        AlgoBase.__init__(self)
        self.k = k

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        # loading vectors of genres for each movie
        ds = MoviesData()
        genres = ds.getGenres()
        years = ds.getYears()

        print("Computing content-based similarity matrix...")

        #building 2d array that serves as a lookup of the similarities between any two movies
        # calculate similarity as 2d matrix
        self.FindSimilarity = np.zeros((self.trainset.n_items, self.trainset.n_items))

        #rating1->movie, rating2->other movie we comparing with
        #looping over the train set and print progress as we process every 100movie
        #e.g: 200 of 1400 movie processed (got similarities)
        for rating1 in range(self.trainset.n_items):
            if (rating1 % 100 == 0):
                print(rating1, " of ", self.trainset.n_items)

            #1- go for next movie(continue looping over trainset)
            #2- convert userId and itemIds to raw innerId(surpriseLib) which used in the predict function
            #3- compute genre and year similarity of every possible pair of movie
            #4- multiply them together to get total similarity score
            #5- assign similarity to similarities obj
            for rating2 in range(rating1+1, self.trainset.n_items):
                movie1Id = int(self.trainset.to_raw_iid(rating1))
                movie2Id = int(self.trainset.to_raw_iid(rating2))

                genreSim = self.getGenreSimilarity(movie1Id, movie2Id, genres)
                yearSim = self.getYearSimilarity(movie1Id, movie2Id, years)

                self.FindSimilarity[rating1, rating2] = genreSim * yearSim
                self.FindSimilarity[rating2, rating1] = self.FindSimilarity[rating1, rating2]
                
        print("Similarity Calculations: Done")
                
        return self

    #cosine similarity between each movie(movie1,movie2) by treating
    #each movie genre as 19-dimensional space that represent all genres
    #going through each genre for both movies to check if they match it or not
    #1- x=genre of movie1, y=genre of movie2
    #2- sumxx = x squared, sumyy = y squared
    #3- sumxy = x*y
    #4- sumxy = sumxy divided by sqrt of sumx*sumy
    def getGenreSimilarity(self, movie1, movie2, genres):
        genres1 = genres[movie1]
        genres2 = genres[movie2]
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(genres1)):
            x = genres1[i]
            y = genres2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
        
        return sumxy/math.sqrt(sumxx*sumyy)


    #using exponential decay function to find rapid decrease between years
    #difference in release years..Note: movies in 80s different than in 90s
    # dividing by 10 (a decade)
    def getYearSimilarity(self, movie1, movie2, years):
        formula = abs(years[movie1] - years[movie2])
        similarity = math.exp(-formula / 10.0)
        return similarity

    #KNN
    #selecting k-nearest movies a user has rated to the one we're predicting
    #based on their release date and genres
    #then find their weighted avg based on their im scores and user ratings

    #takes user id and item id
    #when estimate is called, then it predicts rating of item for a user
    def estimate(self, userId, itemId):
        #user/item exist in dataset?
        if not (self.trainset.knows_user(userId) and self.trainset.knows_item(itemId)):
            raise PredictionImpossible('Item/User not found.')

        # similarity scores between top 40 items user rated and item to be predicted
        totalNeighbors = []
        #loopin for users items (passing u for userid and i for itemid)

        #generating similarity, add it to Similarity variable
        #appending it in neighbor list
        for rating in self.trainset.ur[userId]:
            Similarity = self.FindSimilarity[itemId,rating[0]]
            totalNeighbors.append( (Similarity, rating[1]) )

        # get top k-most similar movies to the movie to be predicted
        topK_Neighbors = heapq.nlargest(self.k, totalNeighbors, key=lambda t: t[0])

        # Compute weighted average of sim score of top K neighbors weighted by user ratings
        totalSimilarity = weightedSum = 0
        for (similarityScore, rating) in topK_Neighbors:
            if (similarityScore > 0):
                totalSimilarity += similarityScore
                weightedSum += similarityScore * rating

        if (totalSimilarity == 0):
            raise PredictionImpossible('No neighbors found')

        predictedRating = weightedSum / totalSimilarity

        return predictedRating
