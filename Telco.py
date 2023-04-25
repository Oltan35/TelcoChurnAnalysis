#########################
# TELCO CHURN ANALYSIS
#########################

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
warnings.simplefilter(action='ignore', category=Warning)

###########################################
# GRAB THE NUMERIC AND CATEGORIC VARIABLE
############################################
df = pd.read_csv("datasets_feature/Telco-Customer-Churn.csv")
df.head()
df.shape

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head())
    print("##################### Tail #####################")
    print(dataframe.tail())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

## GRAB COLS

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)
###################################################
# REALIZE THE TYPE OF THE SOME VARIABLE
###################################################
df.info()
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
####################################################
# NUMERIC AND CATEGORIC VARIABLE ANALYSIS
####################################################
### CAT
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    print(cat_summary(df, col))

### NUM
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    print(num_summary(df, col))

#################################
# TARGET VARIABLE ANALYSIS BY CATEGORIC VARIABLE
#################################
df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)
def cat_cols_target(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

cat_cols_target(df, "Churn", cat_cols)
df.head()
df.info()

#################################
# TARGET VARIABLE ANALYSIS BY NUMERIC VARIABLE
#################################

def num_col_target(dataframe, target, num_cols):
    means = {}
    for col in num_cols:
        means[col] = dataframe.groupby(target)[col].mean()
    return pd.DataFrame(means)


num_col_target(df, "Churn", num_cols)

"""
1. The churn rate of elderly customers is low at 16%.
2. Approximately 25% of the customers have churned.
3. Customers who have monthly contracts have a higher rate of churn.
4. The churn rate of customers who receive technical support is significantly lower than those who do not receive it.
5. Customers who do not have device protection have a higher rate of churn.
6. Customers who do not have online backup have a higher rate of churn.
7. Customers who have online security have a significantly lower churn rate compared to those who do not have it.
8. The churn rate of customers who do not have internet service is lower compared to those who have internet service.
9. Customers who have dependent persons to take care of have a significantly lower churn rate compared to those who do not have any.
"""

###################################
# OUTLIER ANALYSIS
###################################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
low_limit, up_limit = outlier_thresholds(df, num_cols)

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

for col in num_cols:
    grab_outliers(df, col) #10'dan fazla değişken outlier değeri bulunmaktadır.

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    sns.boxplot(df[col])

###################
# MISSING VALUE DETECTION
###################

msno.bar(df)
plt.show()

msno.matrix(df)
plt.show()

msno.heatmap(df)
plt.show()

###################################
# MISSING VALUE HANDLING
###################################

df.isnull().sum()
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False) # Yukarıdan aşağı null col
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

missing_values_table(df, True)

# HANDLE MISSING VALUE
df["TotalCharges"].mean() # 2283.3
df["TotalCharges"].median() # 1397.4
df.groupby("Churn")["TotalCharges"].mean()
# 0   2555.344
# 1   1531.796
df["TotalCharges"] = df["TotalCharges"].fillna(df.groupby("Churn")["TotalCharges"].transform("mean"))
df.groupby("Churn")["TotalCharges"].mean()

###############################################
# FEATURE ENGINEERING
##############################################
df.head()
quantiles = [0.25, 0.5, 0.75, 1]
result = df["tenure"].describe(percentiles=quantiles).T
print(result)
# For Tenure converting the years
df.loc[(df["tenure"] >= 0) & (df["tenure"] <= 12), "TENURE YEAR"] = "0-1 Year"
df.loc[(df["tenure"] > 12) & (df["tenure"] <= 24), "TENURE YEAR"] = "1-2 Years"
df.loc[(df["tenure"] > 24) & (df["tenure"] <= 36), "TENURE YEAR"] = "2-3 Years"
df.loc[(df["tenure"] > 36) & (df["tenure"] <= 48), "TENURE YEAR"] = "3-4 Years"
df.loc[(df["tenure"] > 48) & (df["tenure"] <= 60), "TENURE YEAR"] = "4-5 Years"
df.loc[(df["tenure"] > 60) & (df["tenure"] <= 72), "TENURE YEAR"] = "5-6 Years"

# For Contract
df["Contract"].value_counts()
df["New_Contract"] = df["Contract"].apply(lambda x: 1 if x in ["Two year", "One year"] else 0)

def cat_value(dataframe, col):
    return df[col].value_counts()

for col in cat_cols:
    print(cat_value(df, col))

# For Dependents and Partner
df["New_dep_part"] = df.apply(lambda x: 1 if (x["Partner"] == "Yes") and (x["Dependents"] == "Yes")
                              else 0, axis=1)

# The affect of the extra service purchase

df["Extra_Service"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or
                                              (x["DeviceProtection"] != "Yes") or (x["TechSupport"]
                                               != "Yes") or (x["OnlineSecurity"] != "Yes") else 0, axis=1)

# Young Customers with the new engaged
df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["Contract"] == 0)
                                        and (x["SeniorCitizen"] == 0) else 0, axis=1)

# The one that purchase total service

df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies']] == 'Yes').sum(axis=1)

# Stream Counts
df["Count_Streaming"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

# Payment method

df["Payment_Class"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)", "Credit card (automatic)"] else 0)
df.head()

# Average Monthly Revenue
df["Revenue"] = df["TotalCharges"] / (df["tenure"] + 1)

# New monthly revenue compared to the monthly price
df["New_Mont_Price"] = df["Revenue"] / df["MonthlyCharges"]

# Total service purchasing detect

df["Total_Service_Pricing"] = df["MonthlyCharges"] / (df["NEW_TotalServices"] + 1)

#####################################
# ENCODING PROCESS
#####################################
cat_cols, num_cols, cat_but_car = grab_col_names(df)
df.info()
cat_cols = [col for col in cat_cols if col not in ["NEW_TotalServices", "Churn"]]

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    df = label_encoder(df, col)

df.head()

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols)
df.head()
df.shape
df.corr()
heatmap_df sns.heatmap(df.corr(), cmap="YlGnBu", annot=True) # corr heatmap

##################
# SCALING PROCESSS
##################
scaler = RobustScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

#################
# MODELLING PROCESS
#################
df.head()
y = df["Churn"]
X = df.drop(["Churn","customerID"], axis=1)


models = [('LR', LogisticRegression(random_state=17)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=17)),
          ('RF', RandomForestClassifier(random_state=17)),
          ('XGB', XGBClassifier(random_state=17)),
          ("LightGBM", LGBMClassifier(random_state=17)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=17))]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")
########## LR ##########
# Accuracy: 0.807
# Auc: 0.85
# Recall: 0.5383
# Precision: 0.6709
# F1: 0.5968
########## KNN ##########
# Accuracy: 0.7724
# Auc: 0.7813
# Recall: 0.5302
# Precision: 0.5779
# F1: 0.5529
########## CART ##########
# Accuracy: 0.7257
# Auc: 0.6553
# Recall: 0.5003
# Precision: 0.4835
# F1: 0.4917
########## RF ##########
# Accuracy: 0.7924
# Auc: 0.828
# Recall: 0.4933
# Precision: 0.642
# F1: 0.5577
########## XGB ##########
#Accuracy: 0.7862
#Auc: 0.8258
#Recall: 0.5067
#Precision: 0.6188
#F1: 0.5569
########## LightGBM ##########
#Accuracy: 0.7955
#Auc: 0.8341
#Recall: 0.5217
#Precision: 0.6434
#F1: 0.5757
########## CatBoost ##########
#Accuracy: 0.8001
#Auc: 0.8426
#Recall: 0.519
#Precision: 0.6562
#F1: 0.5795


################################################
# Random Forests
################################################

rf_model = RandomForestClassifier(random_state=17)

rf_params = {"max_depth": [5, 8, None], # Max deep of the tree
             "max_features": [3, 5, 7, "auto"], # Number of features to consider when looking for the best split
             "min_samples_split": [2, 5, 8, 15, 20], # Minimum number of instances required to split a node
             "n_estimators": [100, 200, 500]} # Number of tree

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_ # {'max_depth': None, 'max_features': 7, 'min_samples_split': 15, 'n_estimators': 100}
#rf_final = rf_model.set_params(rf_best_grid.best_params_, random_state=17).fit(X, y)

rf_final = RandomForestClassifier(max_depth=None, max_features=7,min_samples_split=15,n_estimators=100,random_state=17).fit(X, y)
cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1","recall","precision"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_recall'].mean()
cv_results['test_precision'].mean()

# ########## RF ##########
# Base Model
# Accuracy: 0.8009
# Recall: 0.506
# Precision: 0.664
# F1: 0.5744

################################################
# XGBoost
################################################

xgboost_model = XGBClassifier(random_state=17)

xgboost_params = {"learning_rate": [0, 0.01, 1000],
                  "max_depth": [5, 8, 12, 15, 20],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.5, 0.7, 1]}

xgboost_best_grid = RandomizedSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# ########## XGBoost ##########
# Accuracy: 0.794
# F1: 0.565
# Roc-Auc: 0.833

################################################
# LightGBM
################################################

lgbm_model = LGBMClassifier(random_state=17)

lgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# ########## XGBoost ##########
# Accuracy: 0.8022
# F1: 0.582
# Roc-Auc: 0.843

################################################
# CatBoost
################################################

catboost_model = CatBoostClassifier(random_state=17, verbose=False)

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.05, 0.1],
                   "depth": [3, 6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(catboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# ########## CatBoost ##########
# Accuracy:  0.8039
# F1: 0.584
# Roc-Auc: 0.845

################################################
# Feature Importance
################################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X)
plot_importance(xgboost_final, X)
plot_importance(lgbm_final, X)
plot_importance(catboost_final, X)