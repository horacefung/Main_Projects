# ---Basic Methods Class----#
"""
Title:       Basic methods that are frequently used across classes

Description: A class to store methods like pickling files, reading from pickle,
            reusing certain data cleaning methods etc.

Author:      Horace Fung, July 2020
"""

# import packages
import pickle
from sklearn.externals import joblib
import warnings
warnings.filterwarnings('ignore')


class BasicMethods:
    def __init__(self):
        return None

    @staticmethod
    def save_pickle(variable, path):
        output_file = open(path, 'wb')
        pickle.dump(variable, output_file)
        output_file.close()

    @staticmethod
    def read_pickle(path):
        input_file = open(path, 'rb')
        variable = pickle.load(input_file)
        input_file.close()
        return variable

    @staticmethod
    def save_model(model, path):
        joblib.dump(model, path)

    @staticmethod
    def load_model(path):
        model = joblib.load(path)
        return model


if __name__ == "__main__":
    None