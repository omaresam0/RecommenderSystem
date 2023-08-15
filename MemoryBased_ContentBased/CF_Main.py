
from Dataset import MoviesData
from surprise import KNNBasic
from surprise import NormalPredictor
from Recommender import Recommender

import random
import numpy as np

def LoadMoviesData():
    ds = MoviesData()
    print("Getting Ratings...")
    data = ds.loadDataset()
    return (ds, data)

#making sure we get the same results every time
np.random.seed(0)
random.seed(0)

# loading dataset
(ds, data) = LoadMoviesData()

# obj of Recommender recommender
recommender = Recommender(data)

#1-setting similarity parameters
#2-fitting knn basic recommender in surprise lib
#resulting algorithm can be used to retrieve 2x2 of item/user similarity scores


#UserBased CF
UserKNN = KNNBasic(sim_options = {'name': 'pearson_baseline', 'user_based': True})
recommender.AddAlgorithm(UserKNN, "User-Based")


#ItemBased CF
ItemKNN = KNNBasic(sim_options = {'name': 'pearson_baseline', 'user_based': False})
recommender.AddAlgorithm(ItemKNN, "Item-Based")


# Random recommendations (like a baseline)
Random = NormalPredictor()
recommender.AddAlgorithm(Random, "Random")


recommender.Evaluate()

recommender.GetRrecommendations(ds)