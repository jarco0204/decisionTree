import json
import numpy as np
import math

class Attribute(object):

    def __init__(self, attrLabel, dataLabel):
        self.name,self.data = attrLabel,dataLabel
        self.left,self.center,self.right = None,None,None
        self.children = {}


def main():
    #First you gotta load that file
    with open("../data/dataDesc.txt") as f:
        data = json.load(f)

    trainSet = np.loadtxt("../data/train.txt")
    

    ##Calculate Root node
    leftovers = data.copy()
    root, leftovers = findEmptyNode(trainSet, data, leftovers)
    ##Calculate Children
    #leftC, leftovers = findEmptyNode(root.data, data, leftovers)
    #root.left = leftC
    #print(root.data)

    #childrenEntropy
    
    

##Function to calculate entropy
def calcEntropy(dataSet, attr):
    classL = np.fromiter(dataSet[attr], dtype=int)
    lRisk, hRisk = 0,0 ## Remember 1 represents low and 2 represents high
    ##Attr = 0 means we are calculating the general entropy
    if(attr==0):
        for i in classL:
            if i==1:
                lRisk+=1
            else:
                hRisk+=1
        entropy = (-(lRisk/len(classL))* math.log2(lRisk/len(classL)))-(hRisk/len(classL))*math.log2(hRisk/len(classL))
        return entropy
    else:
        labelsattr, count = labelsForAttr(classL)
        labelEntropies = []
        for attr in labelsattr:
            low,high = summary(attr, dataSet[0], classL)
            total = low+high
            labelEntropies.append(-(low/total)*math.log2(low/total)-(high/total)*math.log2(high/total))
        entropy = 0
        index = 0
        for amount in count:
            entropy+= (amount/len(classL))*labelEntropies[index]
            index+=1
        ##print(entropy)
        ##tengo la row, el array 2d de los valores en el row y la cantidad de lows y highs
        return entropy

##
##Finds the best fit in attributes for the current node
##
def findEmptyNode(trainSet, data, leftovers):
    entropySet = calcEntropy(trainSet, 0)
    leftovers.pop(0)
    entropyAtts = [entropySet - calcEntropy(trainSet,i) for i in range(1, len(trainSet))]
    attributeToNode = leftovers[entropyAtts.index(max(entropyAtts))][0]
    leftovers.pop(entropyAtts.index(max(entropyAtts)))
    #print(trainSet)
    indexToConvert = [data.index(i) for i in data if attributeToNode in i]
    in1= [i for i in range(trainSet.shape[1]) if trainSet[indexToConvert[0]][i]==1]
    in1.reverse()
    in2= [i for i in range(trainSet.shape[1]) if trainSet[indexToConvert[0]][i]==2]
    in2.reverse()
    in3= [i for i in range(trainSet.shape[1]) if trainSet[indexToConvert[0]][i]==3]
    in3.reverse()
    leftChildren = trainSet.copy()
    centerChildren = trainSet.copy()
    rightChildren = trainSet.copy()
    for i in in1:
        leftChildren = np.delete(leftChildren, i,axis=1)
    if len(in2)>0:
        for i in in2:
            centerChildren = np.delete(centerChildren, i,axis=1)
    if len(in3)>0:
        for i in in3:
            rightChildren = np.delete(rightChildren, i,axis=1)
    if len(in3) ==0:
        rightChildren = None
    if len(in2) ==0:
        centerChildren = None
    for i in range(len(trainSet)):
        if i in indexToConvert:
            xy = list(trainSet[i])
            x = [0] * (len(xy))
            trainSet[i] = np.fromiter(x, dtype=int)    
    
    return Attribute(attributeToNode, trainSet),leftovers
##
## This function gives you the total low risk, and high risk
## based on the row you give it
## @params label is the row that you want to get a summary for
##          attr is the attribute for what you are lookig specifics
def summary(label, target, attr):
    low,high=0,0
    tar = target.tolist()
    att = attr.tolist()
    #print(len(attr))
    for i in range(len(att)):
        if(att[i]==label):
            if(tar[i]==1):
                low+=1
            else:
                high+=1
    ##print(low, high)
    return low, high
##
## Creates an array specifying how many of each attributes there are
## 
def labelsForAttr(label):
    attrs, count = np.unique(label,return_counts= True)    
    ##print(attrs)
    return attrs, count

if __name__ =='__main__':
    main()