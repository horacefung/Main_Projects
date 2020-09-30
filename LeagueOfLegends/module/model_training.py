# ---Model Training Class----#
"""
Title:       Transformers to perform dimensionality reduction and transformations (e.g log-transforms)
             Perform grid-search to tune top models.

Description: Ingest training data and fit logistic regression, simple classification tree,
            random forest and gradient boosting model.

Author:      Horace Fung, July 2020
"""

import sys

sys.path.append('/Users/horacefung/Documents/GitHub/New_Projects/LeagueOfLegends/module/')

# import packages
# import pandas as pd
import numpy as np
# import pickle
from sklearn import preprocessing
import math
from sklearn.decomposition import PCA
# from sklearn.decomposition import KernelPCA
# from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from basic_methods import BasicMethods  # Inherit basic methods
import pdb

import warnings
warnings.filterwarnings('ignore')


class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, select_features):
        self.select_features = select_features
    
    def fit(self, x_df, y=None):
        return self
    
    def transform(self, x_df, y=None):
        select_features = self.select_features
        x_df = x_df[select_features]
        return(x_df)


class ContinuousFeatureEngineering(BaseEstimator, TransformerMixin):
    
    # Initiate class
    def __init__(self): 
        return None
        
    # We don't need to fit anything, so leave this as is
    def fit(self, x_df, y=None):
        return self
    
    # Perform our feature transformations
    def transform(self, x_df, y=None):
        
        # Log cs field
        add_constant = abs(min(x_df['delta_total_cs']))
        x_df['log_delta_total_cs'] = x_df['delta_total_cs'].apply(lambda x : math.log(x + add_constant + 0.01))
        x_df = x_df.drop('delta_total_cs', axis = 1)
        
        # Create per_level * gamelength variables
        feature_columns = list(x_df.columns)
        per_level = [feature for feature in feature_columns if "perlevel" in feature]
        
        for i in per_level:
            field_name = i + str('_average_gamelength')
            x_df[field_name] = x_df[i] * x_df['average_gamelength']
        
        # Standardize data
        standard_scaler = preprocessing.StandardScaler()
        x_df = standard_scaler.fit_transform(x_df)
        
        return x_df
    

if __name__ == "__main__":

    # if main, parameters
    DATA_DIR = '../data/'
    MODEL_DIR = '../models/'
    XTRAIN_FILE = 'x_train.pkl'
    YTRAIN_FILE = 'y_train.pkl'
    XTEST_FILE = 'x_test.pkl'
    YTEST_FILE = 'y_test.pkl'

    x_train = BasicMethods.read_pickle(DATA_DIR + XTRAIN_FILE)
    y_train = BasicMethods.read_pickle(DATA_DIR + YTRAIN_FILE)
    x_test = BasicMethods.read_pickle(DATA_DIR + XTEST_FILE)
    y_test = BasicMethods.read_pickle(DATA_DIR + YTEST_FILE)

    num_attributes = ['delta_assists', 'delta_damagetochampions', 'delta_deaths',
                      'delta_kills', 'delta_monsterkills', 'delta_total_cs',
                      'delta_totalgold', 'delta_wardskilled', 'delta_wardsplaced',
                      'delta_armor', 'delta_armorperlevel', 'delta_attackdamage',
                      'delta_attackdamageperlevel', 'delta_attackrange', 'delta_attackspeed',
                      'delta_attackspeedperlevel', 'delta_gap_closer_value', 'delta_hard_cc_value', 'delta_hp',
                      'delta_hpperlevel', 'delta_hpregen', 'delta_hpregenperlevel',
                      'delta_movespeed', 'delta_mp', 'delta_mpperlevel', 'delta_mpregen',
                      'delta_mpregenperlevel', 'delta_protection_value',
                      'delta_soft_cc_value', 'delta_spellblock', 'delta_spellblockperlevel',
                      'delta_spells_average_range_value', 'delta_Assassin', 'delta_Fighter',
                      'delta_Mage', 'delta_Marksman', 'delta_Support', 'delta_Tank',
                      'average_gamelength']

    categorical_attributes = ['soul_point', 'red_soul_point']

    # Parameter range determined during training & tuning in jupyter notebooks
    log_param_grid = [

        {'logistic__penalty': ['l1', 'l2'],
         'logistic__C': np.logspace(-4, 4, 20)}
    ]

    cart_param_grid = [
        {'cart__max_depth': [2, 4, 6],
         'cart__min_samples_split': [100, 125, 150],
         'cart__min_samples_leaf': [50, 60, 70]}
    ]

    rf_param_grid = [
        {'rf__max_features': [10, 15, 20],
         'rf__max_depth': [2, 4, 6],
         'rf__min_samples_split': [25, 50, 75],
         'rf__min_samples_leaf': [20, 30, 40]}
    ]

    gbm_param_grid = [
        {'gbm__max_features': [5, 10, 15],
         'gbm__min_samples_split': [1, 10, 15],
         'gbm__min_samples_leaf': [1, 5, 10]}
    ]

    pca_components = 30

    # ----Feature Pipeline ----#
    numerical_pipeline = Pipeline([

        ('FeatureSelector', FeatureSelector(num_attributes)),
        ('FeatureEngineering', ContinuousFeatureEngineering()),
        ('PCA', PCA(n_components=pca_components))
    ])

    categorical_pipeline = Pipeline([

        ('FeatureSelector', FeatureSelector(categorical_attributes))
    ])

    feature_pipeline = FeatureUnion([
        ('numerical_pipeline', numerical_pipeline),
        ('categorical_pipeline', categorical_pipeline)
    ])

    # --- Model Pipelines ----#
    print('Building Model Pipelines')
    log_full_pipeline = Pipeline(steps=[

        ('feature_pipeline', feature_pipeline),
        ('logistic', LogisticRegression())
    ])

    cart_full_pipeline = Pipeline(steps=[

        ('feature_pipeline', feature_pipeline),
        ('cart', DecisionTreeClassifier())
    ])

    rf_full_pipeline = Pipeline(steps=[

        ('feature_pipeline', feature_pipeline),
        ('rf', RandomForestClassifier())
    ])

    gbm_full_pipeline = Pipeline(steps=[

        ('feature_pipeline', feature_pipeline),
        ('gbm', GradientBoostingClassifier())
    ])

    # Fit Models
    print('Grid Search Logistic Regression')
    best_log_model = GridSearchCV(log_full_pipeline, log_param_grid, cv=5, scoring='roc_auc')
    best_log_model.fit(x_train, y_train)

    print('Grid Search CART')
    best_cart_model = GridSearchCV(cart_full_pipeline, cart_param_grid, cv=5, scoring='roc_auc')
    best_cart_model.fit(x_train, y_train)
    # best_cart_model.best_estimator_.named_steps['cart'].feature_importances_
    # pdb.set_trace()

    print('Grid Search Random Forest')
    best_rf_model = GridSearchCV(rf_full_pipeline, rf_param_grid, cv=5, scoring='roc_auc')
    best_rf_model.fit(x_train, y_train)

    print('Grid Search Gradient Boosting')
    best_gbm_model = GridSearchCV(gbm_full_pipeline, gbm_param_grid, cv=5, scoring='roc_auc')
    best_gbm_model.fit(x_train, y_train)

    print(best_log_model)
    print(best_cart_model)
    print(best_rf_model)
    print(best_gbm_model)

    # Get best model test performance
    y_hat_log = best_log_model.predict(x_test)
    y_hat_cart = best_cart_model.predict(x_test)
    y_hat_rf = best_rf_model.predict(x_test)
    y_hat_gbm = best_gbm_model.predict(x_test)

    print('Logistic regression test AUC:', round(roc_auc_score(y_hat_log, y_test), 3))
    print('CART model test AUC:', round(roc_auc_score(y_hat_cart, y_test), 3))
    print('Random forest model test AUC:', round(roc_auc_score(y_hat_rf, y_test), 3))
    print('Gradient boosting model test AUC:', round(roc_auc_score(y_hat_gbm, y_test), 3))

    # Save output
    print('Save Models')
    BasicMethods.save_model(best_log_model, MODEL_DIR + 'best_log_model.pkl')
    BasicMethods.save_model(best_cart_model, MODEL_DIR + 'best_cart_model.pkl')
    BasicMethods.save_model(best_rf_model, MODEL_DIR + 'best_rf_model.pkl')
    BasicMethods.save_model(best_gbm_model, MODEL_DIR + 'best_gbm_model.pkl')
