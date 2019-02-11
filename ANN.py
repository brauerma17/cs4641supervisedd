import csv
from sklearn import tree
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from math import ceil


fn = "phishing_dataset.csv"
writeFn = fn[:-4] + "_ann_output.csv"
print(writeFn)

startTrainPerc = .2
trainPercInc = .10
endTrainPerc = .8


activations = ['tanh']#['identity','logistic','tanh','relu']

solvers = ['lbfgs']

numLayers = [4]

sizeLayers = [25]

layerSlopes = [1]

maxIters = [800]

cvFolds = 2

def runEverything():
    data = preProcessData(fn)
    results = []
    perc = startTrainPerc
    while perc <= endTrainPerc:
        for activation in activations:
            for solver in solvers:
                    for numLayer in numLayers:
                        for sizeLayer in sizeLayers:
                            for layerSlope in layerSlopes:
                                for maxIter in maxIters:
                                    result = [perc,activation,solver,numLayer,sizeLayer,layerSlope,maxIter]
                                    trainError,testError,sumError,diffError = genDecTree(data,perc,activation,solver,numLayer,sizeLayer,layerSlope,maxIter)
                                    valError = crossValTree(data,perc,cvFolds,activation,solver,numLayer,sizeLayer,layerSlope,maxIter)
                                    result.extend([trainError,valError,testError,sumError,diffError])
                                    results.append(result)
                                    print(perc,activation,solver,numLayer,sizeLayer,layerSlope,maxIter)
                    print("Progress")
        perc = perc + trainPercInc


    file = open(writeFn,"w")
    csvW = csv.writer(file)
    csvW.writerow(["Training %","Activation F","Solver","# Layers","Max Layer Size","Layer Slope","Max_Iter","Training Error %","Validation Error %","Testing Error %","Total Error %","Diff in Error %"])
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

def genDecTree(data,perc,theActivation,aSolver,numLayer,sizeLayer,layerSlope,maxIter):
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

    layers = []
    for i in range(0,numLayer):
        layers.append(ceil(sizeLayer/(i+layerSlope)))
    layers = tuple(layers)

    clf = MLPClassifier(solver=aSolver,max_iter=maxIter,hidden_layer_sizes=layers,activation=theActivation)
    clf = clf.fit(x,y)
    predTrainY = clf.predict(x)
    predTestY = clf.predict(tx)

    errorTrain = sum(abs(y - predTrainY))/len(y)
    errorTest = sum(abs(ty - predTestY))/len(ty)
    sumError = (sum(abs(y - predTrainY)) + sum(abs(ty - predTestY)))/(len(y)+len(ty))
    return(errorTrain,errorTest,sumError,abs(errorTrain-errorTest))

    del clf

def crossValTree(data,perc,nFolds,theActivation,aSolver,numLayer,sizeLayer,layerSlope,maxIter):
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

    layers = []
    for i in range(0,numLayer):
        layers.append(ceil(sizeLayer/(i+layerSlope)))
    layers = tuple(layers)

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
        clf = MLPClassifier(solver=aSolver,max_iter=maxIter,hidden_layer_sizes=layers,activation=theActivation)
        clf = clf.fit(trainx,trainy)
        #print(xcuts)
        #print(testx)
        predTestY = clf.predict(testx)
        errorTest = sum(abs(testy - predTestY))/len(testy)
        errors.append(errorTest)

    return (sum(errors)/len(errors))

runEverything()
