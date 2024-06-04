import numpy as np
import pandas as pd 
from sklearn.impute import KNNImputer

def missing_value_imputation(data, n_neighbors=10):
    missing_value_imputer = KNNImputer(n_neighbors=10)
    imputed_data = missing_value_imputer.fit_transform(data)
    return imputed_data