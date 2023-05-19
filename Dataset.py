import csv
import re
from surprise import Dataset
from surprise import Reader

from collections import defaultdict

class MoviesData:

    Id_2_Name = {}
    ratingsPath = 'ratings.csv'
    moviesPath = 'movies.csv'
    
    def loadDataset(self):
        ratings = 0


        #dictionary used to get movieId
        self.Id_2_Name = {}

        #parsing rating dataset [separating columns], skipping first row
        reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)

        #reading rating dataset, loading a custom dataset using builtin class Dataset
        ratings = Dataset.load_from_file(self.ratingsPath, reader=reader)


        #reading movie dataset
        with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
                dataReader = csv.reader(csvfile)
                next(dataReader)  #ignore header row
                for col in dataReader:
                    movieID = int(col[0])
                    movieName = col[1]
                    self.Id_2_Name[movieID] = movieName

        #returning it so it can be used later in recommender func when loadData func is called
        return ratings
    
    def getGenres(self):
        allGenres = defaultdict(list)
        genreIDs = {}
        defaultId = 0
        with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
            dataReader = csv.reader(csvfile)
            next(dataReader)  #ignore header row
            for col in dataReader:
                movieID = int(col[0])
                genreList = col[2].split('|') #list of existing genres
                genreIDList = []
                for genre in genreList: #looping over all genrelist contents
                    if genre in genreIDs:
                        genreID = genreIDs[genre] #put that genre that exists at genreIDs in genreID
                    else:
                        genreID = defaultId
                        genreIDs[genre] = genreID #if genre with value genreID, add to genreIDs dict
                        defaultId += 1
                    genreIDList.append(genreID) #add to genreid list
                allGenres[movieID] = genreIDList #at the end, add genreid list to the allgenres dict under the key of every movie id
        # convert genre lists that encoded in integer to bits so that can be treated as vectors for similarity measuring
        for (movieID, genreIDList) in allGenres.items():
            bits = [0] * defaultId
            for genreID in genreIDList:
                bits[genreID] = 1
            allGenres[movieID] = bits

        return allGenres
    
    def getYears(self):

        allYears = defaultdict(int)
        key = re.compile(r"(?:\((\d{4})\))?\s*$") #to search for 4 word(date)
        with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
            dataReader = csv.reader(csvfile)
            next(dataReader) #Skip header line
            for col in dataReader:
                movieID = int(col[0])
                movieTitle = col[1]
                k = key.search(movieTitle)
                releaseYear = k.group(1)
                if releaseYear:
                    allYears[movieID] = int(releaseYear)
        return allYears

    def getMovieName(self, movieID):
        if movieID in self.Id_2_Name:
            return self.Id_2_Name[movieID]
        else:
            return ""
