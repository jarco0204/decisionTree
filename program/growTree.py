#Modules to be used
import json
import numpy as np
import math

def main():
    #Section loads the file
    trainData= np.loadtxt("../data/train.txt") #Each element of this ndarray is a row
    file = open("../data/dataDesc.txt","r")
    attributeSet = json.load(file)
    file.close()

    #In this new update, the root node is being computed with the same function as for other nodes
    # print("Starting attribute set %s " % attributeSet)
    rootNode= AttributeNode(trainData,attributeSet,None)
    print("Depth 0")
    tryExpandNode(rootNode) #new function to prevent errors when expanding leafs
    print("Depth 1")
    tryExpandNode(rootNode.children[0]) #Expands left child
    print("Depth 2")
    tryExpandNode(rootNode.children[0].children[0]) #Expands left child 
    print("Depth 3")
    tryExpandNode(rootNode.children[0].children[0].children[0]) #Expands left child
    print("Depth 4")
    tryExpandNode(rootNode.children[0].children[0].children[0].children[0]) #Expands left child
    # print("Depth 5")
    # tryExpandNode(rootNode.children[0].children[0].children[0].children[0].children[0]) #Expands left child 
    
    
#Function checks some criteria to determine whether is logical to expand new node
#Makes call to expandNode
#@args node object at a certain depth
#@return None 
def tryExpandNode(node):
    if(checkClassLabel(node)): #This function checks to see whether all objects in the training set have the same class label
            print("Node is not expanded because all elements have same class label")
    else:
        if(len(node.attributes)==2): #first element is class label and second element is the last attribute
            calculateProportionFinalAttribute(node)
            
        else:
            expandNode(node)
            print(node.value)
            print(node.attributes) #This prints the attribute set after removing attribute from node.
            print(node.children[0].data)
        
        
#This function calculates the class label attached to every leaf node in the last attibute
def calculateProportionFinalAttribute(node):
    attrLabel1=[]
    attrLabel2=[]
    attrLabel3=[] # for some nodes this is going to be empty
    i= 0 # this is the index
    allLabelsAr=[]
    for element in node.data[1]: #This is the last attribute
        if(element == 1):
            attrLabel1.append(i)
        elif(element==2):
            attrLabel2.append(i)
        else:
            attrLabel3.append(i)
        i+=1
    allLabelsAr.append(attrLabel1)
    allLabelsAr.append(attrLabel2)
    if(len(attrLabel3)!=0):
        allLabelsAr.append(attrLabel3)

    #Next section looks at the class label associated with each element
    for label in allLabelsAr: #2D array containing index position
        class1=0
        class2=0
        for index in label:
            if(node.data[0][index]==1): #Low risk
                class1+=1
            else:
                class2+=1 #high risk
        #IMPORTANT; as the decision tree algorithm gets better it will be able to predict for when there is 50/50 chance
        #In simple words, if class1 == class2, then just select class 1
        if(class1>= class2):
            node.children.append(1) #This is the class label value (Leaf nodes are just numbers)
        else:
            node.children.append(2)
        
        print("TODO write to file")

            





# #This new function expands the nodes recursively    
# def growTreeRecursively(node):
#     if(node==None): #Base case
#         return node
#     else:
#         expandNode(node)
#         for child in node.children:
#             return growTreeRecursively(child)


#A new function that writes to file when each node is created
def writeNode2File(node):
    a=node.value               #the first value is the value of the selected node
    b=node.children[0].value   #the value of the first child
    c=node.children[1].value   #value of second child
    if len(node.children)==3:  # check for the cases where nodes have 3 branches
        d=node.children[2].value
        e=[a,{1:b,2:c,3:d}]     #Store in the tree as [node,{child1,child2,child3}] if there are 3 children
    else:
        e=[a,{1:b,2:c}]         #Else Store in the tree as [node,{child1,child2}]
    with open("text.txt",'a') as f:
        json.dump(e,f)          #Store tree in text file
    

def checkClassLabel(node):
    yesObj= 0
    noObj= 0
    for element in node.data[0]: #refers to the class label
        if(element==1):
            yesObj+=1
        else:
            noObj+=1
        #Check for early termination
        if((yesObj!=0 and noObj!= 0)):
            return False #There is entropy in the dataset
    return True #All elements share same class label


    

#This function computes the children of a node
#args An AttributeNode object
#returns None
def expandNode(node):
    datasetEntropy= calculateGeneralEntropy(node.data) # This is the entropy value for the whole dataset
    gainAttributes= []
    for attr in range(1,len(node.attributes)): 
        gainAttributes.append(datasetEntropy-calculateEntropyAttribute(node.data,attr))
    arr= np.array(gainAttributes)
    indexMax= np.argmax(arr,axis=0) + 1 # first element contains class label
    node.value=node.attributes[indexMax][0]
    node.attributes.pop(indexMax)
    

    #Next section creates the children of the node based on attribute labels 
    nextArray= buildModifiedTrainData(node.data, indexMax) #Returns a 2D array that classifies according to # of attribute labels
    newAttributes= node.attributes.copy() 
    for array in nextArray:
        
        node.children.append(AttributeNode(array,newAttributes,node))

#This is a helper function that is called for every internal node
#@args the original trainining set and the index[1 to n] of the attribute to classify the dataset
#@returns the modifed array according to the attribute labels [1..3]
def buildModifiedTrainData(trainData,indexMax):
    #This new section creates the next array to work with
    indexesAttr= classifyByAttrLabels(trainData[indexMax]) #indexes used to create new array according to attribute labels
    # print(len(trainData))
    trainData=np.delete(trainData,indexMax,0)
    # print(len(trainData))
    newAr=[]
    for attrLabel in indexesAttr:
        newData=np.zeros(shape=(len(trainData[:,1]),len(attrLabel)))
        i=0
        for element in attrLabel:
            newData[:,i]=trainData[:,element]
            i+=1
        newAr.append(newData)
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
            print("Zero division")
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
        print("this shouldn't execute")
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