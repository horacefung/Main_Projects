import sys

sys.path.append('/Users/horacefung/Documents/GitHub/New_Projects/LeagueOfLegends/module/')

# import packages
import pandas as pd
import requests
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize 
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import pdb
# import os
# import logging
from basic_methods import BasicMethods # Inherit basic methods

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":

    # Change current working directory
    # print(Path(__file__).resolve().parent)

    # if main, parameters
    DATA_DIR = '../data/'
    MODEL_DIR = '../models/'
    MATCH_FILE = '2020_LoL_esports_match_data_from_OraclesElixir_20200722.csv'
    XTRAIN_FILE = 'x_train.pkl'
    YTRAIN_FILE = 'y_train.pkl'
    XTEST_FILE = 'x_test.pkl'
    YTEST_FILE = 'y_test.pkl'
    FULL_FILE = 'full_output.pkl'
    PATCH_END = 10.1  # Up to which patch to use for modeling
    WINDOW = 5
    URL = 'http://ddragon.leagueoflegends.com/cdn/{}/data/en_US/champion/{}.json'

    # cwd = os.getcwd() #Check working directory if needed

    x_train = BasicMethods.read_pickle(DATA_DIR + XTRAIN_FILE)
    full = BasicMethods.read_picskle(DATA_DIR + FULL_FILE)
    pdb.set_trace()








