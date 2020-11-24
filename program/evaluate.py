import numpy as np
import pandas as pd
import json
import math
def main(fname):
    data=np.loadtxt("../data/test.txt")
    data=data.transpose()                                           #Transpose is used to transform the data into columns instead of rows
    columnsData=["RISK","AGE", "CRED_HIS","INCOME","RACE","HEALTH"] # we are using panda to classify the data according to the attribute set
    data=pd.DataFrame(data,columns=columnsData)                     #Add the data to a table
    #print(data)
    with open(r'../data/'+fname,'r') as f:                                   #Open the tree using json
      tree=json.load(f)
    predictions=[]
    for i in range(len(data)):                                      #loop through the test set
      node=tree                                                     # we are using the structure shown in the specification for the project with a little modification
      while node is list:
        v=p[node[0]]
        node=node[1][v]
        node=str(node)
      if (node=='CRED_HIS'):                                        #we check if the attribute set leads to the root
        node=2
      else:
        node=1
      predictions.append(node)
    predictions=np.array(predictions,dtype=float)
    return math.ceil((len(data["RISK"][data["RISK"]==predictions])/len(data))*100)    # we calculate the accuracy 
    
main("fulltree.txt")
