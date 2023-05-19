
from EvaluationMetrics import RecommenderMetrics
from time import perf_counter

#wrapper for all recommender metrics(rmse, mae)
start1 = perf_counter()

class MeasureAccuracy:
    
    def __init__(self, algorithm, AlgoName):
        self.algorithm = algorithm
        self.AlgoName = AlgoName
        
    def Evaluate(self, getData, verbose=True):
        metrics = {}
        # calculate accuracy
        #test function is builtin in algoBase that estimate all the ratings in dataset and returns them in a list of predictions
        #pass perdictions to rec metrics in order to get accuracy


        if (verbose):
            print("Measuring accuracy...")
        self.algorithm.fit(getData.GetTrainSet())
        predictions = self.algorithm.test(getData.GetTestSet())
        metrics["RMSE"] = RecommenderMetrics.RMSE(predictions)
        metrics["MAE"] = RecommenderMetrics.MAE(predictions)

        if (verbose):
            print("Evaluation is done")
            end = perf_counter()
            print(f"Exection time: {end - start1}")
        return metrics

    def GetAlgoName(self):
        return self.AlgoName
    
    def GetAlgorithm(self):
        return self.algorithm

