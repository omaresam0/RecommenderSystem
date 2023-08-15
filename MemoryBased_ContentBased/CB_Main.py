
from Dataset import MoviesData
from ContentBased import ContentBased
from Recommender import Recommender
from surprise import NormalPredictor

import random
import numpy as np

def LoadMoviesData():
    ds = MoviesData()
    print("Getting Ratings...")
    data = ds.loadDataset()
    return (ds, data)

# making sure we get the same results every time
np.random.seed(0)
random.seed(0)

# loading dataset
(ds, data) = LoadMoviesData()

# obj of recommender
recommender = Recommender(data)

#Content Based Filtering
cb = ContentBased()
recommender.AddAlgorithm(cb, "CB-Filtering")

# Random recommendations (like a baseline)
Random = NormalPredictor()
recommender.AddAlgorithm(Random, "Random")

recommender.Evaluate()

recommender.GetRrecommendations(ds)


