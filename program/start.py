#Modules to be used
import json
import numpy as np
import math

def main():
    #Section loads the file
    trainData= np.loadtxt("../data/train.txt") #Each element of this ndarray is a row
    file = open("../data/dataDesc.txt","r")
    m = json.load(file)
    file.close()

    #In this new update, the root node is being computed with the same function as for other nodes
    rootNode= AttributeNode(trainData,m,None)
    expandNode(rootNode) #Depth=0

    #Had this idea of expand in breath-first manner to debug because below code that iterates is not correctling filling the last depth
    #or it is not adding something to denote empty nodes (this is the core problem fidnding a way to halt this)
    
    print("Next Depth")
    expandNode(rootNode.children[0]) # Age is selected
    
    print("Children should be the # of attribute labels: %s" % len(rootNode.children[0].children))
    print()

    print("Next Depth")
    expandNode(rootNode.children[0].children[0]) # Income is selected
    print("Children should be the # of attribute labels: %s" % len(rootNode.children[0].children[0].children))
    print()
    
    print("Next Depth")
    expandNode(rootNode.children[0].children[0].children[0]) # Health is selected
    print("Children should be the # of attribute labels: %s" % len(rootNode.children[0].children))
    #NOTE problem is that for node health number of children node are 3, when at most they can be 2
    

    #Section to compute the next depth iteratively
    # print("Next depth")
    # for node in rootNode.children:
    #     expandNode(node) #Depth=1
    
    # print("New depth")
    # for i in range(len(rootNode.children)):
    #     for node in rootNode.children[i].children:
    #         expandNode(node) #Depth=3

    # print("finalDepth")
    # for i in range(len(rootNode.children)):
    #     for j in range(len(rootNode.children[i].children)):
    #         for node in rootNode.children[i].children[j].children:
    #             expandNode(node)

#TODO
#This new function expands the nodes recursively    
def growTreeRecursively(node):
    if(node==None): #Base case
        return node
    else:
        expandNode(node)
        for child in node.children:
            return growTreeRecursively(child)

#TODO
#A new function that writes to file when each node is created
def writeNode2File():
    pass

#This function computes the children of a node
#args An AttributeNode object
#returns None
#TODO Somewhere we need to check whether all elements share the same class label or not
def expandNode(node):
    datasetEntropy= calculateGeneralEntropy(node.data) # This is the entropy value for the whole dataset
    gainAttributes= []
    print("attributes are: %s" % node.attributes)
    print("This is the data which the node trains with: \n %s " % node.data)
    for attr in range(1,len(node.attributes)): #len(m)
        gainAttributes.append(datasetEntropy-calculateEntropyAttribute(node.data,attr))
    arr= np.array(gainAttributes)
    indexMax= np.argmax(arr,axis=0) + 1 # first element contains class label
    # print("Index max is %s"% indexMax)
    print(" Node is %s" % node.attributes[indexMax][0])

    #Next section creates the children of the node based on attribute labels 
    node.attributes.pop(indexMax)
    print("new attributes are: %s" % node.attributes)
    nextArray= buildModifiedTrainData(node.data, indexMax) #Returns a 2D array 
    for array in nextArray:
        if(len(node.attributes)!=1):
            newAttributes= node.attributes.copy() #Moved to here because I am EAF
            node.children.append(AttributeNode(array,newAttributes,node))
        else:
            node.children.append(None)
            print("No Child ")
            node.value=[0][0]

#This is a helper function that is called for every internal node
#@args the original trainining set and the index[1 to n] of the attribute to classify the dataset
#@returns the modifed array according to the attribute labels [1..3]
#NOTE using print statement, it was verified that this algorithm rebuilds the array based on indexes correctly, might be worth checking it again with some fresh eyes
def buildModifiedTrainData(trainData,indexMax):
    #This new section creates the next array to work with
    indexesAttr= classifyByAttrLabels(trainData[indexMax]) #indexes used to create new array according to attribute labels
    # print("Array of indexes separated based on attribute labels are %s " %indexesAttr)
    trainData=np.delete(trainData,indexMax,0)
    # print(trainData)
    newAr=[]
    for attrLabel in indexesAttr:
        # print("Next attribute label")
        newData=np.zeros(shape=(len(trainData[:,1]),len(attrLabel)))
        # print(newData)
        i=0
        for element in attrLabel:
            newData[:,i]=trainData[:,element]
            i+=1
        newAr.append(newData)
        # print(newData)
    return newAr

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


#This function calculate the entropy for a particular attribute
#@args data which is a simplified dataset and attrIndex which is the index position of attribute [1..5]
#@returns a single value that represents the entropy
def calculateEntropyAttribute(data,attrIndex):
    #Classify each element according to their attribute labels (for this project at most 3)
    row= data[attrIndex]
    attrLabels= classifyByAttrLabels(row)
    
    classSummary=[] #2D array containing the distribution of class label for each attribute label
    for attr in attrLabels:
        lowRisk=0
        highRisk=0
        for element in attr:
            if(data[0][element]==1): #Low risk
                lowRisk+=1
            else: #High risk
                highRisk+=1
        classSummary.append([lowRisk,highRisk])

    #Calculate the entropy for each attribute label
    entropyAttr=[]
    totalsAttr=[]
    for attrLabel in classSummary:
        total= attrLabel[0]+attrLabel[1]
        totalsAttr.append(total)
        try:
            firstTerm= attrLabel[0]/total
            secondTerm= attrLabel[1]/total
        except ZeroDivisionError:
            firstTerm=0
            secondTerm=0

        if(firstTerm!=0 and secondTerm!=0):

            entropyAttrLabel= (-firstTerm * math.log(firstTerm,2))+(-secondTerm * math.log(secondTerm,2))
        else:
            entropyAttrLabel=0
        entropyAttr.append(entropyAttrLabel)
    
    finalEntropyAttribute=0
    i=0

    for attr in entropyAttr:
        if(len(row)!=0):
            finalEntropyAttribute+= (totalsAttr[i]/len(row))*attr
            i+=1
        else:
            finalEntropyAttribute=0

    return finalEntropyAttribute

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
    try:
        firstTerm= lowRisk/len(classLabelRow)
        secondTerm= highRisk/len(classLabelRow)
    except ZeroDivisionError:
        firstTerm=0
        secondTerm=0
    
    if(firstTerm!=0 and secondTerm!=0):
        return(-firstTerm * math.log(firstTerm,2))+(-secondTerm * math.log(secondTerm,2))
    else:
        # print("0")
        return 0
         
#Helper class to Bookkeep the important variables to build each node
class AttributeNode:
    #Constructor
    def __init__(self,dataSet, attributeSet,par):
        self.data= dataSet
        self.attributes= attributeSet
        self.children= [] #array of the AttributeNode children
        self.value= None
        self.parent = par #added back because we need to do it recursively


#Start of the program
if __name__ == "__main__":
    main()