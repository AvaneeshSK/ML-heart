# heart.csv
# https://www.kaggle.com/datasets/arezaei81/heartcsv
    
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import LinearSVC, NuSVC, SVC # benefits from normalizaton, onehotencoding
from sklearn.neighbors import KNeighborsClassifier # benefits from normalizaton, onehotencoding
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

df = pd.read_csv('Machine Learning 2/heart.csv')
df.drop_duplicates(inplace=True)

# correlation matrix
# sns.heatmap(data=df.corr(), annot=True, cmap='Blues')
# plt.show()


# find categories
categories = []
to_be_normalized = []
for col in df.columns:
    if len(df[col].value_counts()) <= 10:
        print(f'{col} : {len(df[col].value_counts())}')
        categories.append(col)
    else:
        to_be_normalized.append(col)

# 0 : female 1 : male
female_heart_disease = 0
female_no_heart_disease = 0
male_heart_disease = 0
male_no_heart_disease = 0
for i in range(len(df)):
    if df['sex'].iloc[i] == 0:
        if df['target'].iloc[i] == 1:
            female_heart_disease += 1
        else:
            female_no_heart_disease += 1
    else:
        if df['target'].iloc[i] == 1:
            male_heart_disease += 1
        else:
            male_no_heart_disease += 1

gender_heart = {
    'no heart disease': [male_no_heart_disease, female_no_heart_disease],
    'heart disease': [male_heart_disease, female_heart_disease]
}

gender_heart_health_df = pd.DataFrame(gender_heart, index=['Male', 'Female'])
plt.style.use('ggplot')
fig, ax = plt.subplots()
bars = gender_heart_health_df.plot.bar(ax=ax)
ax.set_xticklabels(['Male', 'Female'], rotation=0)
ax.set(xlabel='Gender', ylabel='Total')
ax.set_yticks(range(0, max(gender_heart_health_df.values.flatten()+20), 20))
for p in bars.patches:
    ax.annotate(p.get_height(), (p.get_x() + p.get_width() / 2, p.get_height()), xytext=(10, 10), va='center', ha='center', textcoords='offset points')

# onehotencode, normalize (preprocessing)
df_to_normalize = df.copy()
ct = ColumnTransformer([
    ('minmax', MinMaxScaler(), to_be_normalized),
    ('onehot', OneHotEncoder(sparse_output=False), categories[:-1])
], remainder='passthrough')

transformed_data = ct.fit_transform(df_to_normalize)
df_to_normalize = pd.DataFrame(transformed_data)

X_norm_1hotenc = df_to_normalize.drop(columns=[30])
y_norm_1hotenc = df_to_normalize[30]

X_train_norm1hotenc, X_test_norm1hotenc, y_train_norm1hotenc, y_test_norm1hotenc = train_test_split(X_norm_1hotenc, y_norm_1hotenc, test_size=0.2, random_state=42)

# without any preprocessing
X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# modelling : (all models)

# linearSVC
linearSVC_model_1 = LinearSVC(dual=False)
linearSVC_model_1.fit(X_train_norm1hotenc, y_train_norm1hotenc)
linearSVC_model_1_score = linearSVC_model_1.score(X_test_norm1hotenc, y_test_norm1hotenc)
linearSVC_model_2 = LinearSVC(dual=False)
linearSVC_model_2.fit(X_train, y_train)
linearSVC_model_2_score = linearSVC_model_2.score(X_test, y_test)

# SVC
SVC_model_1 = SVC()
SVC_model_1.fit(X_train_norm1hotenc, y_train_norm1hotenc)
SVC_model_1_score = SVC_model_1.score(X_test_norm1hotenc, y_test_norm1hotenc)
SVC_model_2 = SVC()
SVC_model_2.fit(X_train, y_train)
SVC_model_2_score = SVC_model_2.score(X_test, y_test)

# NuSVC
NuSVC_model_1 = NuSVC()
NuSVC_model_1.fit(X_train_norm1hotenc, y_train_norm1hotenc)
NuSVC_model_1_score = NuSVC_model_1.score(X_test_norm1hotenc, y_test_norm1hotenc)
NuSVC_model_2 = NuSVC()
NuSVC_model_2.fit(X_train, y_train)
NuSVC_model_2_score = NuSVC_model_2.score(X_test, y_test)

# KNeighborsClassifier
KNeighborsClassifier_model_1 = KNeighborsClassifier()
KNeighborsClassifier_model_1.fit(X_train_norm1hotenc, y_train_norm1hotenc)
KNeighborsClassifier_model_1_score = KNeighborsClassifier_model_1.score(X_test_norm1hotenc, y_test_norm1hotenc)
KNeighborsClassifier_model_2 = KNeighborsClassifier()
KNeighborsClassifier_model_2.fit(X_train, y_train)
KNeighborsClassifier_model_2_score = KNeighborsClassifier_model_2.score(X_test, y_test)

# LogisticRegression
LogisticRegression_model_1 = LogisticRegression(solver='newton-cholesky')
LogisticRegression_model_1.fit(X_train_norm1hotenc, y_train_norm1hotenc)
LogisticRegression_model_1_score = LogisticRegression_model_1.score(X_test_norm1hotenc, y_test_norm1hotenc)
LogisticRegression_model_2 = LogisticRegression(solver='newton-cholesky')
LogisticRegression_model_2.fit(X_train, y_train)
LogisticRegression_model_2_score = LogisticRegression_model_2.score(X_test, y_test)

# RandomForestClassifier
RandomForestClassifier_model_1 = RandomForestClassifier()
RandomForestClassifier_model_1.fit(X_train_norm1hotenc, y_train_norm1hotenc)
RandomForestClassifier_model_1_score = RandomForestClassifier_model_1.score(X_test_norm1hotenc, y_test_norm1hotenc)
RandomForestClassifier_model_2 = RandomForestClassifier()
RandomForestClassifier_model_2.fit(X_train, y_train)
RandomForestClassifier_model_2_score = RandomForestClassifier_model_2.score(X_test, y_test)

scores_dict = {
    'preprocessed' : [linearSVC_model_1_score, SVC_model_1_score, NuSVC_model_1_score, KNeighborsClassifier_model_1_score, LogisticRegression_model_1_score, RandomForestClassifier_model_1_score],
    'raw' : [linearSVC_model_2_score, SVC_model_2_score, NuSVC_model_2_score, KNeighborsClassifier_model_2_score, LogisticRegression_model_2_score, RandomForestClassifier_model_2_score]
}
scores_dict2 = {
    'linearSVC' : [linearSVC_model_1_score, linearSVC_model_2_score],
    'SVC': [SVC_model_1_score, SVC_model_2_score],
    'NuSVC': [NuSVC_model_1_score, NuSVC_model_2_score],
    'KNeighborsClassifier': [KNeighborsClassifier_model_1_score, KNeighborsClassifier_model_2_score],
    'LogisticRegression': [LogisticRegression_model_1_score, LogisticRegression_model_2_score],
    'ForestClassifier': [RandomForestClassifier_model_1_score, RandomForestClassifier_model_2_score]
}

scores_df = pd.DataFrame(scores_dict2, index=['Preprocessed', 'Raw'])
fig2, ax2 = plt.subplots()
scores_df.plot.bar(ax=ax2)


# randomized search

# randomized_forest_1 = RandomizedSearchCV(
#     cv=5,
#     estimator=RandomForestClassifier(),
#     n_iter=1000,
#     n_jobs=-1,
#     verbose=2,
#     param_distributions={
#         'n_estimators': [50, 100, 150, 200, 300, 400, 500, 600, 1000, 1200, 1500],
#         'criterion': ['gini', 'entropy', 'log_loss'],
#         'max_depth': [0, 1, 2, 3, 4, 5, 6, 10, 15, 20, 30, 40, 50, 100, 200, 500, 1000],
#         'min_samples_split': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 80, 120, 500],
#         'min_samples_leaf': [1, 2, 3, 4, 5, 10, 20, 40, 100],
#         'max_features': [None, 'sqrt', 'log2'],
#         'max_leaf_nodes': [None, 1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 100, 120, 200, 300, 500],
#         'bootstrap': [False, True],
#         'warm_start': [False, True],
#         'max_samples': [None, 1, 2, 3, 4, 8, 10, 20, 30, 60, 100, 120, 200, 500, 1000]
#     }
# )
# randomized_forest_2 = RandomizedSearchCV(
#     cv=5,
#     estimator=RandomForestClassifier(),
#     n_iter=1000,
#     n_jobs=-1,
#     verbose=2,
#     param_distributions={
#         'n_estimators': [50, 100, 150, 200, 300, 400, 500, 600, 1000, 1200, 1500],
#         'criterion': ['gini', 'entropy', 'log_loss'],
#         'max_depth': [0, 1, 2, 3, 4, 5, 6, 10, 15, 20, 30, 40, 50, 100, 200, 500, 1000],
#         'min_samples_split': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 80, 120, 500],
#         'min_samples_leaf': [1, 2, 3, 4, 5, 10, 20, 40, 100],
#         'max_features': [None, 'sqrt', 'log2'],
#         'max_leaf_nodes': [None, 1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 100, 120, 200, 300, 500],
#         'bootstrap': [False, True],
#         'warm_start': [False, True],
#         'max_samples': [None, 1, 2, 3, 4, 8, 10, 20, 30, 60, 100, 120, 200, 500, 1000]
#     }
# )
# randomized_forest_1.fit(X_train_norm1hotenc, y_train_norm1hotenc)
# randomized_forest_2.fit(X_train, y_train)

# print(randomized_forest_1.best_params_)
# print(randomized_forest_2.best_params_)

# grid search

# gridsearch_forest_1 = GridSearchCV(
#     cv=5,
#     estimator=RandomForestClassifier(),
#     n_jobs=-1,
#     verbose=2,
#     param_grid={
#         'n_estimators': [300, 350, 400],
#         'criterion': ['entropy', 'log_loss'],
#         'max_depth': [20, 25, 35, 40],
#         'min_samples_split': [5, 6, 7, 8, 9, 10],
#         'min_samples_leaf': [1, 2],
#         'max_features': ['log2'],
#         'max_leaf_nodes': [3, 5, 6, 25, 30, 35],
#         'bootstrap': [True],
#         'warm_start': [True],
#         'max_samples': [15, 70, 80, 90, 100]
#     }
# )
# gridsearch_forest_2 = GridSearchCV(
#     cv=5,
#     estimator=RandomForestClassifier(),
#     n_jobs=-1,
#     verbose=2,
#     param_grid={
#         'n_estimators': [300, 350, 400],
#         'criterion': ['entropy', 'log_loss'],
#         'max_depth': [20, 25, 35, 40],
#         'min_samples_split': [5, 6, 7, 8, 9, 10],
#         'min_samples_leaf': [1, 2],
#         'max_features': ['log2'],
#         'max_leaf_nodes': [3, 5, 6, 25, 30, 35],
#         'bootstrap': [True],
#         'warm_start': [True],
#         'max_samples': [15, 70, 80, 90, 100]
#     }
# )
# gridsearch_forest_1.fit(X_train_norm1hotenc, y_train_norm1hotenc)
# gridsearch_forest_2.fit(X_train, y_train)
# print(gridsearch_forest_1.best_params_)
# print(gridsearch_forest_2.best_params_)

# gridsearch_forest_1_score = gridsearch_forest_1.score(X_test_norm1hotenc, y_test_norm1hotenc)
# gridsearch_forest_2_score = gridsearch_forest_2.score(X_test, y_test)

# fig, ax3 = plt.subplots()
# ax3.bar(['processed', 'raw'], [gridsearch_forest_1_score, gridsearch_forest_2_score])
# plt.show()

# found best parameters:
# {'bootstrap': True, 'criterion': 'log_loss', 'max_depth': 35, 'max_features': 'log2', 'max_leaf_nodes': 3, 'max_samples': 90, 'min_samples_leaf': 2, 'min_samples_split': 8, 'n_estimators': 300, 'warm_start': True}
# {'bootstrap': True, 'criterion': 'entropy', 'max_depth': 25, 'max_features': 'log2', 'max_leaf_nodes': 3, 'max_samples': 80, 'min_samples_leaf': 2, 'min_samples_split': 6, 'n_estimators': 300, 'warm_start': True}



# experiment with best parameters vs default on RANDOMFORESTCLASSIFIER

forest_default_1 = RandomForestClassifier()
forest_default_2 = RandomForestClassifier()
forest_tuned_1 = RandomForestClassifier(
    bootstrap=True,
    criterion='log_loss',
    max_depth=35,
    max_features='log2',
    max_leaf_nodes=3,
    max_samples=90,
    min_samples_leaf=2,
    min_samples_split=8,
    n_estimators=300,
    warm_start=True
)
forest_tuned_2 = RandomForestClassifier(
    bootstrap=True,
    criterion='entropy',
    max_depth=25,
    max_features='log2',
    max_leaf_nodes=3,
    max_samples=80,
    min_samples_leaf=2,
    min_samples_split=6,
    n_estimators=300,
    warm_start=True
)

forest_default_1.fit(X_train_norm1hotenc, y_train_norm1hotenc)
forest_default_2.fit(X_train, y_train)
forest_tuned_1.fit(X_train_norm1hotenc, y_train_norm1hotenc)
forest_tuned_2.fit(X_train, y_train)

forest_default_1_score=  forest_default_1.score(X_test_norm1hotenc, y_test_norm1hotenc)
forest_default_2_score=  forest_default_2.score(X_test, y_test)
forest_tuned_1_score=  forest_tuned_1.score(X_test_norm1hotenc, y_test_norm1hotenc)
forest_tuned_2_score=  forest_tuned_2.score(X_test, y_test)

forest_scores = {
    'Default Forest': [forest_default_1_score, forest_default_2_score],
    'Tuned Forest': [forest_tuned_1_score, forest_tuned_2_score]
}
forest_scores_df = pd.DataFrame(forest_scores, index=['Preprocessed', 'Raw'])

fig, ax4 = plt.subplots()
forest_scores_df.plot.bar(ax=ax4)
plt.show()
