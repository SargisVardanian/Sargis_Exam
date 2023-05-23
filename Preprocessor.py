import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# from sklearn.manifold import TSNE
# from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


def drop_data(df, target_col_name= None, eps=0.1):
    corr_matrix = df.corr()
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    target_corr_series = corr_matrix[target_col_name]
    low_corr_cols = target_corr_series[abs(target_corr_series) < eps].index.tolist()
    cols_to_drop = obj_cols + low_corr_cols
    return  df.drop(cols_to_drop, axis=1)

# nan_columns = df.columns[df.isna().any()].tolist()
#
# if nan_columns:
#     print('Столбцы с NaN значениями:', nan_columns)
# else:
#     print('Все столбцы не содержат NaN значения')
def polynom_data(df, target, eps=0.2):
    df = drop_data(df, target)
    X = df.drop(target, axis=1)
    y = df[target]
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly.fit_transform(X)

    df_poly = pd.DataFrame(X_poly, columns=poly.get_feature_names_out())
    df_poly[target] = y
    df_poly = drop_data(df_poly, target, eps)
    return df_poly


def remove_outliers(data, target_column, n_components=2, outlier_threshold=3):
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    outliers = (X_pca[:, 0] < -outlier_threshold) | (X_pca[:, 0] > outlier_threshold) | \
               (X_pca[:, 1] < -outlier_threshold) | (X_pca[:, 1] > outlier_threshold)

    X_filtered = X[~outliers]
    y_filtered = y[~outliers]

    return X_filtered, y_filtered



df = pd.read_csv('Albert_spotify_target popularity.csv')
df = polynom_data(df, target='popularity', eps=0.25)
X = df.drop('popularity', axis=1)
y = df['popularity']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.8)


x_train.to_csv('dataset_poly_train.csv', index=False)
y_train.to_csv('dataset_poly_train_labels.csv', index=False)

x_test.to_csv('dataset_poly_test.csv', index=False)
y_test.to_csv('dataset_poly_test_labels.csv', index=False)