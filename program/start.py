#Modules to be used
import json
import numpy as np
import math

def main():
    trainData= np.loadtxt("../data/train.txt") #Each element of this ndarray is a row
    file = open("../data/dataDesc.txt","r")
    m = json.load(file)
    file.close() 
    
    attributeSet, newDataSet= findRootNode(trainData,m)
    for newData in newDataSet:
        datasetEntropy= calculateGeneralEntropy(newData) # This is the entropy value for the whole dataset
        gainAttributes= []
        for attr in range(1,len(attributeSet)): #len(m)
            gainAttributes.append(datasetEntropy-calculateEntropyAttribute(newData,attr))
        arr= np.array(gainAttributes)
        indexMax= np.argmax(arr,axis=0) + 1 # first element contains class label
        print(" Node is %s" % m[indexMax][0])
        attributeSet.pop(indexMax)
    
    


    

def findRootNode(trainData,m):
    datasetEntropy= calculateGeneralEntropy(trainData) # This is the entropy value for the whole dataset
    gainAttributes= []
    for attr in range(1,len(m)): #len(m)
        gainAttributes.append(datasetEntropy-calculateEntropyAttribute(trainData,attr))
    arr= np.array(gainAttributes)
    indexMax= np.argmax(arr,axis=0) + 1 # first element contains class label
    print("Root Node is %s" % m[indexMax][0])
    m.pop(indexMax)

    newAr= buildModifiedTrainData(trainData,indexMax)
    
    return m, newAr

#This is a helper function that is called for every internal node
#@args the original trainining set and the index[1 to n] of the attribute to classify the dataset
#@returns the modifed array according to the attribute labels [1..3]
def buildModifiedTrainData(trainData,indexMax):
    #This new section creates the next array to work with
    indexesAttr= classifyByAttrLabels(trainData[indexMax]) #indexes used to create new array according to attribute labels
    trainData=np.delete(trainData,indexMax,0)
    newAr=[]
    for attrLabel in indexesAttr:
        newData=np.zeros(shape=(len(trainData[:,1]),len(attrLabel)))
        i=0
        for element in attrLabel:
            newData[:,i]=trainData[:,element]
            i+=1
        newAr.append(newData)
    return newAr

#This function calculate the entropy of the whole dataset, in order to find the gain for each attribute.
#@args data which is a ndnumpy array
#@return the entropy for the dataset
def calculateGeneralEntropy(data):
    classLabelRow= data[0] #First Row contains the class label entries
    lowRisk=0
    highRisk=0
    for element in classLabelRow:
        if(element==1): #Low Risk
            lowRisk+=1
        else:
            highRisk+=1
    firstTerm= lowRisk/len(classLabelRow)
    secondTerm= highRisk/len(classLabelRow)
    return (-firstTerm * math.log(firstTerm,2))+(-secondTerm * math.log(secondTerm,2))
    


def calculateEntropyAttribute(data,attrIndex):

    #Classify each element according to their attribute class (for this project at most 3)
    row= data[attrIndex]
    attrLabels= classifyByAttrLabels(row)

    classSummary=[]
    for attr in attrLabels:
        lowRisk=0
        highRisk=0
        for element in attr:
            if(data[0][element]==1): #Low risk
                lowRisk+=1
            else: #High risk
                highRisk+=1
        classSummary.append([lowRisk,highRisk])
    print(classSummary)

    #Calculate the entropy for each attribute label
    entropyAttr=[]
    totalsAttr=[]
    for attrLabel in classSummary:
        total= attrLabel[0]+attrLabel[1]
        totalsAttr.append(total)
        firstTerm= attrLabel[0]/total
        secondTerm= attrLabel[1]/total
        if(firstTerm!=0 and secondTerm!=0):
            entropyAttrLabel= (-firstTerm * math.log(firstTerm,2))+(-secondTerm * math.log(secondTerm,2))
        else:
            entropyAttrLabel=0
        entropyAttr.append(entropyAttrLabel)
    
    finalEntropyAttribute=0
    i=0
    for attr in entropyAttr:
        finalEntropyAttribute+= (totalsAttr[i]/len(row))*attr
        i+=1

    return finalEntropyAttribute

#This function classifies an attribute/row according to their labels
#@args takes an array containing the attribute to classify
#@return a 2D array containing the indexes of the class label
def classifyByAttrLabels(row):
    attrLabels=[[],[],[]] #stores indexes for each attribute label
    for element in range(len(row)):
        if(row[element]==1):
            attrLabels[0].append(element)
        elif(row[element]==2):
            attrLabels[1].append(element)
        elif(row[element]==3):
            attrLabels[2].append(element)

    if(len(attrLabels[2])==0): #Some attributes don't have 3 labels
        attrLabels.pop(2)
    return attrLabels

        
        
    


    



main()