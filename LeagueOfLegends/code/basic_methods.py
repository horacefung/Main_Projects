##---Basic Methods Class----#
"""
Title:       Basic methods that are frequently used across classes

Description: A class to store methods like pickling files, reading from pickle,
			 reusing certain data cleaning methods etc.

Author:      Horace Fung, July 2020
"""

#import packages
import pandas as pd
import json
import pickle
import numpy as np
import pdb
import os

import warnings
warnings.filterwarnings('ignore')


class BasicMethods():


    def __int__(self):
        return(None)

     @staticmethod
    def save_pickle(variable, path):

        output_file = open(path,'wb')
        pickle.dump(x_train, output_file)
        output.close()

    @staticmethod
    def read_pickle(path):

        input_file = open(path,'rb')
        variable = pickle.load(input_file)
        input_file.close()
        return(variable)