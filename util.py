from scipy import stats
import numpy as np
from math import log
import random

# This method computes entropy for information gain
def entropy(class_y):

    if len(class_y) == 0:
        return 0
     
    
    
    y = [k[-1] for k in class_y]
    y = np.array(y, dtype = int)
    total = len(y)
    num1 = sum(i for i in y)
    num0 = total - num1
    entropy = 0
    p1 = float(num1) / total
    p0 = float(num0) / total
    if p1 != 0. and p0 != 0.:
        entropy = -p1 * log(p1, 2) - p0 * log(p0, 2)
    else:
        entropy = 0
    return entropy


def partition_classes(X, y, split_attribute, split_val):
    
    X_left = []
    X_right = []
    
    y_left = []
    y_right = []
    
    if isinstance(split_val,float) or isinstance(split_val,int):
        for i in range(len(X)):
            if X[i][split_attribute] <= split_val:
                X_left.append(X[i])
                y_left.append(y[i])
            else:
                X_right.append(X[i])
                y_right.append(y[i])
    else:
        for i in range(len(X)):
            if X[i][split_attribute] == split_val:
                X_left.append(X[i])
                y_left.append(y[i])
            else:
                X_right.append(X[i])
                y_right.append(y[i])
    return (X_left, X_right, y_left, y_right)

    
def information_gain(previous_y, current_y):
    fraction0 = 1.0 * len(current_y[0]) / len(previous_y)
    fraction1 = 1.0 - fraction0

    old_entropy = entropy(previous_y)
    new_entropy = entropy(current_y[0])*fraction0 + entropy(current_y[1])*fraction1

    info_gain = old_entropy - new_entropy

    return info_gain


    
