import csv
from sklearn import tree
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import KFold


fn = "phishing_dataset.csv"
writeFn = fn[:-4] + "_tree_output.csv"
print(writeFn)

startTrainPerc = .30
trainPercInc = .10
endTrainPerc = .80

startMaxDepth = 2
endMaxDepth = 10
maxDepthInc = 1

#min sample split will decrement, min val is 2
startMinSampSplit = 8
endMinSampSplit = 2
minSampSplitInc = 1

cvFolds = 3

def runEverything():
    data = preProcessData(fn)
    results = []
    maxDepth = startMaxDepth
    while maxDepth <= endMaxDepth:
        perc = startTrainPerc
        while perc <= endTrainPerc:
            minSampSplit = startMinSampSplit
            while minSampSplit >= endMinSampSplit:
                result = [perc,maxDepth,minSampSplit]
                trainError,testError,sumError,diffError = genDecTree(data,perc,maxDepth,minSampSplit)
                valError = crossValTree(data,perc,cvFolds,maxDepth,minSampSplit)
                result.extend([trainError,valError,testError,sumError,diffError])
                results.append(result)
                minSampSplit = minSampSplit - minSampSplitInc
            perc = perc + trainPercInc
        maxDepth = maxDepth + maxDepthInc
    #print(results)

    file = open(writeFn,"w")
    csvW = csv.writer(file)
    csvW.writerow(["Training %","Max Depth","Min Sample Split","Training Error %","Validation Error %","Testing Error %","Total Error %","Diff in Error %"])
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

def genDecTree(data,perc,maxDepth=None,min_samp_split=2,min_samp_leaf=1,max_num_leaf=None):
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

    clf = tree.DecisionTreeClassifier(criterion="entropy",max_depth=maxDepth,min_samples_split=min_samp_split)
    clf = clf.fit(x,y)
    predTrainY = clf.predict(x)
    predTestY = clf.predict(tx)

##    print(clf)
##    print("Trained on", trainPerc * 100, "%")
##    print("testing this algorithm on training set now - ")
##    print("incorrectly classified",sum(abs(y - predTrainY)),"out of",len(y), "instances")
##    print("Percent error on train set:", sum(abs(y - predTrainY))/len(y)*100 , "%")
##    print("")
##    print("testing this algorithm on testing set now - ")
##    print("incorrectly classified",sum(abs(ty - predTestY)), "out of", len(ty), "instances")
##    print("Percent error on test set:", sum(abs(ty - predTestY))/len(ty)*100 , "%")
##    print("")
##    print("")

    errorTrain = sum(abs(y - predTrainY))/len(y)
    errorTest = sum(abs(ty - predTestY))/len(ty)
    sumError = (sum(abs(y - predTrainY)) + sum(abs(ty - predTestY)))/(len(y)+len(ty))
    return(errorTrain,errorTest,sumError,abs(errorTrain-errorTest))

    del clf

def crossValTree(data,perc,nFolds,maxDepth=None,min_samp_split=2,min_samp_leaf=1,max_num_leaf=None):
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
        clf = tree.DecisionTreeClassifier(criterion="entropy",max_depth=maxDepth,min_samples_split=min_samp_split)
        clf = clf.fit(trainx,trainy)
        #print(xcuts)
        #print(testx)
        predTestY = clf.predict(testx)
        errorTest = sum(abs(testy - predTestY))/len(testy)
        errors.append(errorTest)

    return (sum(errors)/len(errors))

runEverything()
