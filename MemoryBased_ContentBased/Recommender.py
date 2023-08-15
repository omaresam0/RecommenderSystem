from time import perf_counter

from BuildData import BuildData
from MeasureAccuracy import MeasureAccuracy
from MeasureAccuracy import start1
class Recommender:
    
    algorithms = []

    def __init__(self, dataset):
        data = BuildData(dataset)
        self.dataset = data

     #taking algorithm simOption and algorithm name 'string' from main class in order to evaluate it
     #1-pass it to MeasureAccuracy that will use it in its evaluate function
     #2-add accuracy results are added to an array

    def AddAlgorithm(self, algorithm, name):
        alg = MeasureAccuracy(algorithm, name)
        self.algorithms.append(alg)

    #1-create result array
    #2-loop on the alg array and check for each algorithm(Evaluating + algorithmName)
    #3-evaluate function of measureaccuracy is called to evaluate each algorithm in the results dict
    #in results dict: key=algorithm, value=accuracy results
    def Evaluate(self):
        results = {}
        for algorithm in self.algorithms:
            print("Evaluating", algorithm.GetAlgoName(), "...")
            results[algorithm.GetAlgoName()] = algorithm.Evaluate(self.dataset)
            #AlgName + RMSE + MAE
        print("\n")

        # printing accuracy results
        #done by looping on results array and getting accuracy saved in there
        print("{:<10} {:<10} {:<10}".format("Algorithm", "RMSE", "MAE"))
        for (AlgoName, metrics) in results.items():
            print("{:<10} {:<10.4f} {:<10.4f}".format(AlgoName, metrics["RMSE"], metrics["MAE"]))


    #testUser is the userId(e.g: user15)
    #looping on algorithms array
    #convert movie IDs into titles
    #this function only to print the movie recommendations that are generated; called in main class.
    def GetRrecommendations(self, ml, testUser=15):

        for algo in self.algorithms:
            print("\nUsing Algorithm:", algo.GetAlgoName())

            #training the algorithm using full complete training test
            #since in this function we only want best recommendations
            #but in evaluation phase we must split data to know the correct accuracy

            print("\nBuilding model...")
            trainSet = self.dataset.GetFullTrainSet()
            algo.GetAlgorithm().fit(trainSet)

            #1- get all movies that user hasn't rated before
            #2- process the test set of the algorithm which give a set of rating predictions
            #3- generate rating predictions
            print("Getting Recommendations...")
            testSet = self.dataset.GetAntiTestSet(testUser)
            predictions = algo.GetAlgorithm().test(testSet)

            recommendations = []

            print ("\nRecommendations for you:")
            for userID, movieID, actualRating, expectedRating, _ in predictions:
                movie_Id = int(movieID)
                recommendations.append((movie_Id, expectedRating))

            recommendations.sort(key=lambda x: x[1], reverse=True)
            #generate 10 movies
            for ratings in recommendations[:10]:
                print(ml.getMovieName(ratings[0]), ratings[1])
            end = perf_counter()
            print(f"Exection time: {end - start1}")