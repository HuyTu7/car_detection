import pandas as pd
import json
import numpy
import csv
import networkx as nx
import operator
import random
from matplotlib import pyplot

def get_labels():
    labels = dict()
    for i in range(550):
        if i < 500:
            labels["neg-%s.pgm" % i] = 0
        labels["pos-%s.pgm" % i] = 1
    with open('car_labels.json', 'w') as fp:
        json.dump(labels, fp)
    return labels



if __name__ == '__main__':
    print get_labels()
    '''
    with open('school_ids_test.json') as file1:
        s_ids = json.load(file1, encoding="UTF-8")
    '''