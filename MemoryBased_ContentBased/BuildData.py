
from surprise.model_selection import train_test_split

class BuildData:
    
    def __init__(self, data):

        #Build a full training set to evaluate overall features/used at predictions step
        self.TrainSet = data.build_full_trainset()

        #Build a 75% train & 25% test split
        self.trainSet, self.testSet = train_test_split(data, test_size=.25, random_state=1)

    #used for recommendations (in AntiTestSet)
    def GetFullTrainSet(self):
        return self.TrainSet

    #movies that user hasn't rated before
    #antitestset = any item in trainSet but not in user items
    #rawId is normal id specifiec in dataset, while innerId is a unique integer id that surprise lib can manipulate
    def GetAntiTestSet(self, testUser):
        trainset = self.TrainSet
        avg = trainset.global_mean
        anti_testset = []
        userId = trainset.to_inner_uid(str(testUser))
        user_items = set([j for (j, _) in trainset.ur[userId]])
        anti_testset += [(trainset.to_raw_uid(userId), trainset.to_raw_iid(i), avg) for
                                 i in trainset.all_items() if
                                 i not in user_items]
        return anti_testset

    def GetTrainSet(self):
        return self.trainSet
    
    def GetTestSet(self):
        return self.testSet
