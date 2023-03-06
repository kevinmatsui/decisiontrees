import numpy as np
import pandas as pd
import math


class GenTree:
    def __init__(self, label = None,children = None,value = None, depth = None):  
        self.label = label        
        self.children = children if children is not None else []
        self.value = value
        self.depth = depth     

def finishGuess(data, features):
    vals = pd.unique(data["label"])
    frequency = data["label"].value_counts()
    idx = frequency.idxmax()
    return idx

class DecisionTree:
    dtree: GenTree
    deepest: int
    def __init__(self):
        self.dtree = GenTree(depth = -1)
        self.deepest = -1
        pass



    def id3Wrap(self, data,features,maxDepth):
        
        def id3(data,features,tree):  
            if maxDepth-1 <= tree.depth:
                tree.label = "label"
                tree.depth += 1
                if tree.depth > self.deepest:
                    self.deepest = tree.depth
                tree.children.append(GenTree(value= finishGuess(data,features), depth = tree.depth)) 
                return
            # if there is only one label
            uniqueLabels = pd.unique(data["label"])
            if len(uniqueLabels) == 1:
                tree.depth +=1
                if tree.depth > self.deepest:
                    self.deepest = tree.depth
                tree.label = "label"
                # add one child from tree, that is the one label
                tree.children.append(GenTree(value=uniqueLabels[0], depth=tree.depth))
                return
            newRoot = findRoot(data, features)
            tree.label = newRoot
            tree.depth+=1
            if tree.depth > self.deepest:
                self.deepest = tree.depth
            for val in np.unique(data[newRoot]):       
                recurTree = GenTree(value=val, depth = tree.depth)
                tree.children.append(recurTree)
                id3(data[data[newRoot] == val], features, recurTree)

        id3(data,features, self.dtree)

    def predictWrap(self, row):
        def predictionRec(tree, row):
            #print(len(tree.children))
            if len(tree.children) == 1:
                #print("SSSSSS"
                if tree.children[0].value != None:
                    return tree.children[0].value
                else:
                    return "unacc"
            # the label of tree is one of the features that can be found in the column
            feature1 = tree.label
            for kid in tree.children:
                if kid.value == row[feature1]:
                    return predictionRec(kid, row)

        tree = self.dtree
        return predictionRec(tree, row)
def entropy(column):
    vals, counts = np.unique(column, return_counts=True)
    entropy = np.sum(
        [-(counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) if counts[i] > 0 else 0 for i in
         range(len(counts))])
    print("entropy for given data: ", entropy)
    return entropy


def informationGain(data, featureName):
    vals, counts = np.unique(data[featureName], return_counts=True)
    origEntropy = entropy(data["label"])
    featureEntp = np.sum(
        [counts[i] / np.sum(counts) * entropy(data.loc[data[featureName] == vals[i], "label"]) for i in
         range(len(counts))])
    return origEntropy - featureEntp


def findRoot(data, features):
    largest = 0
    nextRoot = ""
    for feature in features:
        gain = informationGain(data, feature)
        if gain > largest:
            largest = gain
            nextRoot = feature
    print("The Root Feature Is: " + nextRoot)
    print("The Information Gain Is: " ,largest)
    return nextRoot


    

traindata = pd.read_csv("train.csv")
testdata = pd.read_csv("test.csv")
features = ["buying","maint","doors","persons","lug_boot","safety"]

dedtree = DecisionTree()
dedtree.id3Wrap(traindata,features,10)
preder = []
for l in range(len(traindata)):
    guess = dedtree.predictWrap(traindata.iloc[l])
    preder.append(guess)
chek = traindata["label"]
counter = 0
for s in range(len(traindata)):
    if chek.iloc[s] == preder[s]:
        counter+=1
print("On the training data, the accuracy of the decision tree in decimal is: ",counter/len(traindata))

checker = testdata["label"]
counter = 0
predict = []
for x in range(len(testdata)):
    guess = dedtree.predictWrap(testdata.iloc[x])
    predict.append(guess)
for y in range(len(testdata)):
    if checker.iloc[y] == predict[y]:
        counter+=1
print("On the testing data, the accuracy of the decision tree in decimal is: ",counter/len(testdata))
# for x in range(len(traindata)):
#     guess = decTreeTr.predictWrap(traindata.iloc[x])
#     if guess != None:
#         predictions.append(guess)
#     else:
#         predictions.append("unacc")

# expected = traindata["label"]
# correct = 0
# for x in range(len(traindata)):
#     if expected.iloc[x] == predictions[x]:
#         correct+=1
# print("On the training data, the accuracy of the decision tree in decimal is: ",correct/len(traindata))



#print(predictions)
print("The Max Depth of the Tree is : ", dedtree.deepest)
foldDataSet = list()
fold1 = pd.read_csv("fold1.csv")
fold2 = pd.read_csv("fold2.csv")
fold3 = pd.read_csv("fold3.csv")
fold4 = pd.read_csv("fold4.csv")
fold5 = pd.read_csv("fold5.csv")

avgstd =[]
avgmaxdepths = []
for b in range(5):
    avg = 0
    for x in range(5):
        if x == 0:
            train = pd.concat([fold2,fold3,fold4,fold5], axis = 0)
            test = fold1
        if x == 1:
            train = pd.concat([fold3,fold4,fold5,fold1], axis=0)
            test = fold2
        if x == 2:
            train = pd.concat([fold4,fold5,fold1, fold2], axis=0)
            test = fold3
        if x == 3:
            train = pd.concat([fold5,fold1,fold2,fold3], axis=0)
            test = fold4
        if x == 4:
            train = pd.concat([fold1,fold2,fold3,fold4], axis=0)
            test = fold5
        dedtree = DecisionTree()
        dedtree.id3Wrap(train,features,b+1)
        preder = []

        for y in range(len(test)):
            guess = dedtree.predictWrap(test.iloc[y])
            preder.append(guess)
        actual = test["label"]
        count = 0
        for j in range(len(actual)):
            if actual.iloc[j] == preder[j]:
                count+=1
        print("The accuracy in decimal of the number " + str(x+1) + " run of the 5-fold cross validation, with maxDepth of " + str(b+1)+ " is : ", count/len(actual))
        avg+=count/len(actual)
        avgstd.append(count/len(actual))
    avg = avg/5
    print("The average in decimal of the accuracy of the 5-fold cross validations, with maxDepth of " + str(b+1)+ " is : ", avg)
    avgmaxdepths.append(avg)
print("The five different max depths's average cross validation accuracy shown: ")
print(avgmaxdepths)

firstSTD=math.sqrt(((avgstd[0] - avgmaxdepths[0])**2 + (avgstd[1] - avgmaxdepths[0])**2 +(avgstd[2] - avgmaxdepths[0])**2 +(avgstd[3] - avgmaxdepths[0])**2 + (avgstd[4] - avgmaxdepths[0])**2 ) / 5)
secondSTD=math.sqrt(((avgstd[5] - avgmaxdepths[1])**2 + (avgstd[6] - avgmaxdepths[1])**2 +(avgstd[7] - avgmaxdepths[1])**2 +(avgstd[8] - avgmaxdepths[1])**2 + (avgstd[9] - avgmaxdepths[1])**2 ) / 5)
thirdSTD=math.sqrt(((avgstd[10] - avgmaxdepths[2])**2 + (avgstd[11] - avgmaxdepths[2])**2 +(avgstd[12] - avgmaxdepths[2])**2 +(avgstd[13] - avgmaxdepths[2])**2 + (avgstd[14] - avgmaxdepths[2])**2 ) / 5)
fourthSTD=math.sqrt(((avgstd[15] - avgmaxdepths[3])**2 + (avgstd[16] - avgmaxdepths[3])**2 +(avgstd[17] - avgmaxdepths[3])**2 +(avgstd[18] - avgmaxdepths[3])**2 + (avgstd[19] - avgmaxdepths[3])**2 ) / 5)
fifthSTD=math.sqrt(((avgstd[20] - avgmaxdepths[4])**2 + (avgstd[21] - avgmaxdepths[4])**2 +(avgstd[22] - avgmaxdepths[4])**2 +(avgstd[23] - avgmaxdepths[4])**2 + (avgstd[24] - avgmaxdepths[4])**2 ) / 5)

stds = [firstSTD, secondSTD, thirdSTD, fourthSTD, fifthSTD]
print("The five different max depth's standard deviation, from the cross validation average is shown:")
print(stds)
print("The Best Depth Is 5")

dedtree = DecisionTree()
dedtree.id3Wrap(traindata,features,5)
preder = []
for l in range(len(traindata)):
    guess = dedtree.predictWrap(traindata.iloc[l])
    preder.append(guess)
chek = traindata["label"]
counter = 0
for s in range(len(traindata)):
    if chek.iloc[s] == preder[s]:
        counter+=1
print("On the training data, the accuracy of the depth 5 limited tree is : ",counter/len(traindata))
preder = []
for l in range(len(testdata)):
    guess = dedtree.predictWrap(testdata.iloc[l])
    preder.append(guess)
chek = testdata["label"]
counter = 0
for s in range(len(testdata)):
    if chek.iloc[s] == preder[s]:
        counter+=1
print("On the testing data, the accuracy of the depth 5 limited tree is : ",counter/len(testdata))
print(dedtree.deepest)