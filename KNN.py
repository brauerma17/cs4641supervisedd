import csv
from sklearn import tree
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from math import ceil
from sklearn.neighbors import KNeighborsClassifier


fn = "phishing_dataset.csv"
writeFn = fn[:-4] + "_knn_output.csv"
print(writeFn)

startTrainPerc = .30
trainPercInc = .10
endTrainPerc = .80

ks = [1,3,5,7,9,11,13,15,17,19,21]

weightings = ['uniform','distance']

ps = [1,2]

cvFolds = 3

def runEverything():
    data = preProcessData(fn)
    results = []
    perc = startTrainPerc
    while perc <= endTrainPerc:
        for p in ps:
            for weighting in weightings:
                    for k in ks:
                        result = [perc,p,weighting,k]
                        trainError,testError,sumError,diffError = genDecTree(data,perc,p,weighting,k)
                        valError = crossValTree(data,perc,cvFolds,p,weighting,k)
                        result.extend([trainError,valError,testError,sumError,diffError])
                        results.append(result)
                    print("Progress")
        perc = perc + trainPercInc


    file = open(writeFn,"w")
    csvW = csv.writer(file)
    csvW.writerow(["Training %","Distance Alg","Nearest Alg","k","Training Error %","Validation Error %","Testing Error %","Total Error %","Diff in Error %"])
    csvW.writerows(results)
    print("Open your file")



def preProcessData(fn):
    data = []
    f = open(fn)

    csvReader = csv.reader(f,delimiter=",")
    for row in csvReader:
        data.append(row)

    data = data[1:]
    i = 0
    for col0 in data[0]:
        try:
            float(col0)
            for row in data:
                row[i] = float(row[i])
        except:
            col = []
            for row in data:
                col.append(row[i])
            le = preprocessing.LabelEncoder()
            le.fit(col)
            a = le.transform(col)
            newCol = np.array(a).tolist()
            #print(newCol)
            for row in data:
                #print(i)
                row[i] = newCol[i]
        i = i + 1

    return data

def genDecTree(data,perc,aP,weighting,k):
    trainPerc = perc
    train = []
    x = []
    y = []
    c = 0
    while c <=trainPerc*(len(data)):
        train.append(data[c])
        x.append(data[c][:-1])
        y.append(data[c][-1])
        c = c + 1

    #print("Train set ends on row " + str(c))
    test = []
    tx = []
    ty = []
    while c < len(data):
        test.append(data[c])
        tx.append(data[c][:-1])
        ty.append(data[c][-1])
        c = c + 1

    clf = KNeighborsClassifier(n_neighbors=k,weights=weighting,p=aP)
    clf = clf.fit(x,y)
    predTrainY = clf.predict(x)
    predTestY = clf.predict(tx)

    errorTrain = sum(abs(y - predTrainY))/len(y)
    errorTest = sum(abs(ty - predTestY))/len(ty)
    sumError = (sum(abs(y - predTrainY)) + sum(abs(ty - predTestY)))/(len(y)+len(ty))
    return(errorTrain,errorTest,sumError,abs(errorTrain-errorTest))

    del clf

def crossValTree(data,perc,nFolds,aP,weighting,k):
    trainPerc = perc
    work = []
    x = []
    y = []
    c = 0
    while c <=trainPerc*(len(data)):
        work.append(data[c])
        x.append(data[c][:-1])
        y.append(data[c][-1])
        c = c + 1

    numRows = c - 1
    sizeCut = numRows//nFolds

    cuts = []
    xcuts = []
    ycuts = []
    for i in range(0,(numRows//sizeCut)):
        xcut = x[(i*sizeCut):(i+1)*sizeCut]
        ycut = y[(i*sizeCut):(i+1)*sizeCut]
        xcuts.append(xcut)
        ycuts.append(ycut)
        #print(xcut,"\n\n\n")

    errors = []
    for i in range(0,(numRows//sizeCut)):
        testx = xcuts[i]
        testy = ycuts[i]
        trainx = []
        trainy = []
        for j in range(0,(numRows//sizeCut)):
            if j!=i:
                trainx.extend(xcuts[j])
                trainy.extend(ycuts[j])
        clf = KNeighborsClassifier(n_neighbors=k,weights=weighting,p=aP)
        clf = clf.fit(trainx,trainy)
        #print(xcuts)
        #print(testx)
        predTestY = clf.predict(testx)
        errorTest = sum(abs(testy - predTestY))/len(testy)
        errors.append(errorTest)

    return (sum(errors)/len(errors))

runEverything()
