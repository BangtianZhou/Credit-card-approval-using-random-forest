from util import entropy, information_gain, partition_classes
import numpy as np
import ast
import random








class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        #self.tree = []
        self.tree = {}
        pass
    
    def majorityCnt(self,quality):
        total=len(quality)
        num_1=sum(quality)
        num_0=total-num_1
        if num_1>=num_0:
            return 1
        else:
            return 0

    def creatTree(self, X, y, depth, indice):
        nrow=len(X)
        ncol=len(y)
        quality = [i for i in y]
        depth += 1
        if ncol == 1:
            return self.majorityCnt(quality)
        elif nrow <=30:
            return self.majorityCnt(quality)
        elif depth > 4:
            return self.majorityCnt(quality)
        elif len(quality)==sum(quality):
            return 1
        elif sum(quality)==0:
            return 0
        else:
            bestFeat,bestVal = self.select_feature(X,y,indice)
            tree = {bestFeat:{}}
            subX1,subX2,subY1,subY2 = partition_classes(X, y, bestFeat, bestVal)
            if isinstance(bestVal,float) or isinstance(bestVal,int):
                tree[bestFeat]["<=" + str(bestVal)] = self.creatTree(subX1, subY1, depth, indice)
                tree[bestFeat][">" + str(bestVal)] = self.creatTree(subX2, subY2, depth, indice)
            else:
                tree[bestFeat]["==" + str(bestVal)] = self.creatTree(subX1, subY1, depth, indice)
                tree[bestFeat]["!=" + str(bestVal)] = self.creatTree(subX2, subY2, depth, indice)
        
            return tree

    
    
    
    
    
    
    
    def select_feature(self,X,y,indice):
        dataset = np.c_[X,y]
        baseEntropy = entropy(dataset)
        choose_infoGain = 0.0
        bestFeature = -1

        for i in indice:
            vals = [example[i] for example in X]
            univals = sorted(set(vals))
            newEntropy = 0.0
            #for value in univals:
            c = 0
            #value = random.choice(univals)
            #bestValue = value
            while c <10:
                c+=1
                value = random.choice(univals)
                #bestValue = value
                subX1,subX2,subY1,subY2 = partition_classes(X, y, i, value)
                p1 = len(subY1)/float(len(X))
                p2 = len(subY2)/float(len(X))
                subdataset1 = np.c_[subX1,subY1]
                subdataset2 = np.c_[subX2,subY2]
                newEntropy = p1 * entropy(subdataset1) + p2 * entropy(subdataset2)
                infoGain = baseEntropy - newEntropy
                #print(infoGain,choose_infoGain)
                if (infoGain >= choose_infoGain):
                    choose_infoGain = infoGain
                    bestFeature = i
                    bestValue = value

        return bestFeature,bestValue
    
    
    
    
    
    
    def learn(self, X, y):
        # Train the decision tree (self.tree) using the the sample X and labels y
        # print(y)
        numFeatures = len(X[0])
        
        indice = np.random.choice(numFeatures, size=2, replace=False)
        
        self.tree = self.creatTree(X,y,0,indice)
        
        
        
        
        pass
    
    
    def classify(self, record):
        # TODO: classify the record using self.tree and return the predicted label
        
        
        result = 0
        myTree = self.tree
        while type(myTree)==dict:
            index=myTree.keys()[0]
            value = record[index]
            myTree = myTree[index]
            myValueStr = myTree.keys()[0]
            
            #print(myValueStr)
            
            
            if ">" in myValueStr:
                myValue = myValueStr[1:]
                if value <= myValue:
                    myTree = myTree[myTree.keys()[1]]
                else:
                    myTree = myTree[myTree.keys()[0]]
            elif "<=" in myValueStr:
                myValue = myValueStr[2:]
                if value <= myValue:
                    myTree = myTree[myTree.keys()[0]]
                else:
                    myTree = myTree[myTree.keys()[1]]
            elif "==" in myValueStr:
                myValue = myValueStr[1:]
                if value == myValue:
                    myTree = myTree[myTree.keys()[1]]
                else:
                    myTree = myTree[myTree.keys()[0]]
            else:
                myValue = myValueStr[2:]
                if value == myValue:
                    myTree = myTree[myTree.keys()[0]]
                else:
                    myTree = myTree[myTree.keys()[1]]
            # print(myTree)
        result = int(myTree)
        return result
        
        
        pass
