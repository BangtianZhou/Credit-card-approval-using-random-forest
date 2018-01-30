from decision_tree import DecisionTree
import csv
import numpy as np  # http://www.numpy.org
import ast
import random



number = 653             #change this line for different size of data


class RandomForest(object):
    num_trees = 0
    decision_trees = []

    # the bootstrapping datasets for trees
    # bootstraps_datasets is a list of lists, where each list in bootstraps_datasets is a bootstrapped dataset.
    bootstraps_datasets = []

    # the true class labels, corresponding to records in the bootstrapping datasets
    # bootstraps_labels is a list of lists, where the 'i'th list contains the labels corresponding to records in 
    # the 'i'th bootstrapped dataset.
    bootstraps_labels = []

    def __init__(self, num_trees):
        # Initialization done here
        self.num_trees = num_trees
        self.decision_trees = [DecisionTree() for i in range(num_trees)]


    def _bootstrapping(self, XX, n):
        index = []
        feature = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        for i in range(0,number):
            index.append(random.randint(1, number - 2))

        samples = [] # sampled dataset
        labels = []  # class labels for the sampled records

        for i in range(0,number):
            temp = []
            #print(i)
            for j in range(0,16):
                #print(index[i],j)
                temp.append(XX[index[i]][feature[j]])
            samples.append(temp)
        labels = [n[-1] for n in samples]
        samples = [n[:-1] for n in samples]
        #print labels
        return (samples, labels)


    def bootstrapping(self, XX):
        # Initializing the bootstap datasets for each tree
        for i in range(self.num_trees):
            data_sample, data_label = self._bootstrapping(XX, len(XX))
            self.bootstraps_datasets.append(data_sample)
            self.bootstraps_labels.append(data_label)


    def fitting(self):
        for i in range(self.num_trees):
            new_X = []
            new_y = []
            
            for n in range(1,number):
                tem = int(np.random.random_integers(0,number-3))
                
                new_X.append(self.bootstraps_datasets[0][tem])
                new_y.append(self.bootstraps_labels[0][tem])
                    #print(new_y)
            self.decision_trees[i].learn(new_X,new_y)
            # print indice
            


        pass


    def voting(self, X):
        y = []
        
        for record in X:
            # Following steps have been performed here:
            #   1. Find the set of trees that consider the record as an 
            #      out-of-bag sample.
            #   2. Predict the label using each of the above found trees.
            #   3. Use majority vote to find the final label for this recod.
            votes = []
            for i in range(len(self.bootstraps_datasets)):
                dataset = self.bootstraps_datasets[i]
                if record not in dataset:
                    OOB_tree = self.decision_trees[i]
                    effective_vote = OOB_tree.classify(record)
                    votes.append(effective_vote)

            # print(votes)
            counts = np.bincount(votes)
            
            if len(counts) == 0:
                y = np.append(y,1000)
                pass
            else:
                y = np.append(y, np.argmax(counts))

        return y

def main():
    X = list()
    y = list()
    XX = list()  # Contains data features and data labels
    numerical_cols = set([1, 2, 7, 10, 13, 14, 15]) # indices of numeric attributes (columns)

    # Loading data set
    print 'reading data'
    with open("data.csv") as f:
        next(f, None)

        for line in csv.reader(f, delimiter=","):
            xline = []
            for i in range(len(line)):
                if i in numerical_cols:
                    xline.append(ast.literal_eval(line[i]))
                else:
                    xline.append(line[i])

            X.append(xline[:-1])
            y.append(xline[-1])
            XX.append(xline[:])

    # VERY IMPORTANT: Minimum forest_size should be 10
    forest_size = 20
    
    # Initializing a random forest.
    randomForest = RandomForest(forest_size)

    # Creating the bootstrapping datasets
    print 'creating the bootstrap datasets'
    randomForest.bootstrapping(XX)

    # Building trees in the forest
    print 'fitting the forest'
    randomForest.fitting()

    # Calculating an unbiased error estimation of the random forest
    # based on out-of-bag (OOB) error estimate.
    y_predicted = randomForest.voting(X)

    # Comparing predicted and true labels
    results = [prediction == truth for prediction, truth in zip(y_predicted, y)]

    # Accuracy
    accuracy = float(results.count(True)) / float(len(results))

    print "accuracy: %.4f" % accuracy
    print "OOB estimate: %.4f" % (1-accuracy)


if __name__ == "__main__":
    main()
