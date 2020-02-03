# California Housing price prediction

# Importing libraries
import os
import pdb
import hashlib
import tarfile
from six.moves import urllib
import numpy as np
import pandas as pd
from zlib import crc32
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Getting data
DATA_SOURCE = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
PATH = os.path.join(os.getcwd(), "Data/")
DATA_URL = DATA_SOURCE + "datasets/housing/housing.tgz"


# def fetch_data(url, path):
#     pdb.set_trace()
#     if not os.path.isdir(path):
#         os.mkdir(path)
#     tgz_path = os.path.join(path, "housing.tgz")
#     urllib.request.urlretrieve(url, tgz_path)
#     data_tgz = tarfile.open(tgz_path)
#     data_tgz.extractall(path=path)
#     data_tgz.close()
#
# fetch_data(url=DATA_URL, path=PATH)

# Loading data
def load_data(path):
    csv_path = os.path.join(path, "housing.csv")
    return pd.read_csv(csv_path)


data = load_data(PATH)
data['ocean_proximity'].value_counts()
data.describe()
# data.hist(bins=50, figsize=(20, 15));plt.show()

#train, test = train_test_split(data, test_size=0.2, random_state=42)

# Stratified sampling based on the income category
data["income_cat"] = pd.cut(data["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
#data["income_cat"].hist();plt.show()
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_ind, test_ind in split.split(data, data["income_cat"]):
    strat_train = data.loc[train_ind]
    strat_test = data.loc[test_ind]

#strat_test["income_cat"].value_counts()/len(strat_test)
for set_ in (strat_train, strat_test):
    set_.drop("income_cat", axis=1, inplace=True)

# Data visulization
# Visualizing geographic data
train1 = strat_train.copy()
train1.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.show()
train1.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=train1["population"]/100, label="population", figsize=(10, 7), c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True, )
plt.legend()
plt.show()


# Correlation amongst attributes
corr_matrix = train1.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(train1[attributes], figsize=(12, 8))
plt.show()

# Ploting most correlated attributes
train1.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
plt.axis([0, 16, 0, 550000])
plt.show()

# Experimenting with combination of different attributes
train1["rooms_per_household"] = train1["total_rooms"]/train1["households"]
train1["bedrooms_per_room"] = train1["total_bedrooms"]/train1["total_rooms"]
train1["population_per_household"] = train1["population"]/train1["households"]

# Checking correlation
corr_matrix = train1.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# Data preparing for machine learning model
train1 = strat_train.drop("median_house_value", axis=1)
train1_label = strat_train["median_house_value"].copy()

# sample_incomplete_rows = train1[train1.isnull().any(axis=1)].head()
# sample_incomplete_rows


# Feature scaling and transformation pipeline
from custom_attributes import CombinedAttributesAdder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


#attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
#train1_extra_attribs = attr_adder.transform(train1.values)

pdb.set_trace()
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])


#train1_num_tr = num_pipeline.fit_transform(train1_num)


from sklearn.compose import ColumnTransformer

num_attribs = list(train1.drop("ocean_proximity", axis=1))
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs)
])

train1_prepared = full_pipeline.fit_transform(train1)

# ------ Model training --------
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# ------ Linear Regression -------
lin_reg = LinearRegression()
lin_reg.fit(train1_prepared, train1_label)
# ------ Descision Tree Regressor ------
tree_reg = DecisionTreeRegressor()
tree_reg.fit(train1_prepared, train1_label)

# let's try the full preprocessing pipeline on a few training instances
some_data = train1.iloc[:5]
some_labels = train1_label.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))

# -------- Measuring errors -------
from sklearn.metrics import mean_squared_error
#train1_pred = lin_reg.predict(train1_prepared)
train1_pred = tree_reg.predict(train1_prepared)
lin_mse = mean_squared_error(train1_label, train1_pred)
lin_rmse = np.sqrt(lin_mse)
print("RMSE for LR: {}".format(lin_rmse))

# ------ Cross validation scores -------
from sklearn.model_selection import cross_val_score
# ---- Scoring for Descision Tree
scores = cross_val_score(tree_reg, train1_prepared, train1_label, scoring='neg_mean_squared_error', cv=10)
tree_rmse_score = np.sqrt(-scores)
print("Scores:", tree_rmse_score)
print("Mean:", tree_rmse_score.mean())
print("Standard deviation:", tree_rmse_score.std())

# ---- Scoring for Linear Regression
lin_scores = cross_val_score(lin_reg, train1_prepared, train1_label, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
print("Scores:", lin_rmse_scores)
print("Mean:", lin_rmse_scores.mean())
print("Standard deviation:", lin_rmse_scores.std())

#-------- Random Forest Regressor --------
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(train1_prepared, train1_label)
# Training scores
train1_pred1 = forest_reg.predict(train1_prepared)
mse = mean_squared_error(train1_label, train1_pred1)
rmse = np.sqrt(mse)
print("Training RMSE for RF: {}".format(rmse))

# Cross Validation scores (10-fold CV)
forest_scores = cross_val_score(forest_reg, train1_prepared, train1_label, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
print("Scores:", forest_rmse_scores)
print("Mean:", forest_rmse_scores.mean())
print("Standard deviation:", forest_rmse_scores.std())

# #------ Saving model -------
# # Save
# from sklearn.externals import joblib
# joblib.dump(my_model, "my_model.pkl")
# # Load
# my_model_loaded = joblib.load("my_model.pkl")
# #----------------------------

# ---------- Fine Tuning-Grid Search CV ---------
from sklearn.model_selection import GridSearchCV

param_grid = [
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True)
grid_search.fit(train1_prepared, train1_label)
grid_search.best_params_
grid_search.best_estimator_

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

# Relative importance of each attribute for making accurate predictions
feature_importances = grid_search.best_estimator_.feature_importances_
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)

# -------- Evaluating model on test set --------
final_model = grid_search.best_estimator_

X_test = strat_test.drop("median_house_value", axis=1)
y_test = strat_test["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_pred = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_pred)
final_rmse = np.sqrt(final_mse)

# ---- 95% confidence interval -----
from scipy import stats
confidence = 0.95
squared_errors = (final_pred - y_test)**2
confi_interval = np.sqrt(stats.t.interval(confidence, len(squared_errors)-1, loc=squared_errors.mean(), scale=stats.sem(squared_errors)))

pdb.set_trace()