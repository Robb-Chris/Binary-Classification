# Modules
# install '! pip install imbalanced-learn'
import itertools
import random
import numpy as np
import pandas as pd
from re import X
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import  SVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from imblearn.over_sampling import RandomOverSampler

# Data Preparation
data = pd.read_csv('data.csv')

# One-Hot Encode 'Customer_Type' Column
encoder = OneHotEncoder()
encoder_df = pd.DataFrame(encoder.fit_transform(data[['Customer_Type']]).toarray())
data = data.join(encoder_df)
data.drop('Customer_Type', axis = 1, inplace = True)
data = data.rename(columns = {0 : 'New_Customer', 1 : 'Other', 2 : 'Returning_Customer'})

# Column list for concatenation
column_names = list(data.columns.values)
column_names.pop(0)

x_data = data.drop('Transaction',axis = 1).values
y_data = data['Transaction'].values

# Randomly oversample the minority class
ros = RandomOverSampler(random_state = 42)
x_ros, y_ros= ros.fit_resample(x_data, y_data)

# Principal Component Analysis & Dim Reduction
# Scaling/Normalizing Data
scaler = StandardScaler()
x_data_scaled = scaler.fit_transform(x_ros)
# Perform PCA
pca = PCA(n_components=None)
pca.fit(x_data_scaled)
# explain at least 90% of the variance
explained_variances = pca.explained_variance_ratio_
total_variance = 0
n_components = 0
for explained_variance in explained_variances:
    total_variance += explained_variance
    n_components += 1
    
    if total_variance >= 0.9:
        break
pca = PCA(n_components=n_components)
x_data_pca = pca.fit_transform(x_data_scaled)

# Feature Selection with mutual info classification
mi_scores = mutual_info_classif(x_ros, y_ros.ravel(), random_state = 42)
high_score_features = []
selector = SelectKBest(mutual_info_classif, k=5)
selected_features = selector.fit_transform(x_ros, y_ros.ravel())
selected_feature_indices = selector.get_support(indices=True)
x_data_fs = x_ros[:, selected_feature_indices]

# Train-Validation-Test Split
# Full dataset with oversampling
x_train, x_test, y_train, y_test = train_test_split(x_ros, y_ros, test_size=0.2, random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1)
# Dataset with feature selection
x_train_fs, x_test_fs, y_train, y_test = train_test_split(x_data_fs, y_ros, test_size=0.2, random_state=1)
x_train_fs, x_val_fs, y_train, y_val = train_test_split(x_train_fs, y_train, test_size=0.25, random_state=1)
# Dataset with PCA
x_train_pca, x_test_pca, y_train, y_test = train_test_split(x_data_pca, y_ros, test_size=0.2, random_state=1)
x_train_pca, x_val_pca, y_train, y_val = train_test_split(x_train_pca, y_train, test_size=0.25, random_state=1)


# Indivudual Algorithms with cross-validation hyperparameter tuning
def RF(dataset):
    if dataset == 'fs':
        X_train = x_train_fs
        X_val = x_val_fs
        X_test = x_test_fs
    if dataset == 'pca':
        X_train = x_train_pca
        X_val = x_val_pca
        X_test = x_test_pca   
    else:
        X_train = x_train
        X_val = x_val
        X_test = x_test

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10]
    }

    # Create a random forest classifier
    rf = RandomForestClassifier(random_state=42)

    # Create a grid search object with cross validation (number of cross-validation folds cv=5)
    grid_search = GridSearchCV(rf, param_grid, cv=5, verbose=0)

    # Fit the grid search object on the training data and use cross validation for evaluation
    grid_search.fit(X_train, y_train)

    # Access the best hyperparameters and the best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Evaluate the best model on the validation set
    y_val_pred = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    # Evaluate the best model on the test set
    y_test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    class_report = classification_report(y_test, y_test_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_test_pred)

    # print(f'Best Parameters {data}:', best_params)
    # print('Validation Accuracy:', val_accuracy)
    # print('Test Accuracy:', test_accuracy, '\n')
    # print(classification_report(y_test, y_test_pred))
    # print(confusion_matrix(y_test, y_test_pred))

    info_dict = {
        'algorithm': 'RF',
        'dataset': f'{dataset}',
        'parameter': best_params,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
        'f1-score': class_report['macro avg']['f1-score'],
        'conf_matrix': conf_matrix
    }

    return info_dict

def SVM(dataset):
    if dataset == 'fs':
        X_train = x_train_fs
        X_val = x_val_fs
        X_test = x_test_fs
    if dataset == 'pca':
        X_train = x_train_pca
        X_val = x_val_pca
        X_test = x_test_pca   
    else:
        X_train = x_train
        X_val = x_val
        X_test = x_test

    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [0.01, 0.1, 1, 5],
        'kernel': ['rbf', 'sigmoid']
    }

    # Create SVM classifier
    svm = SVC(random_state=42) 

    # Create a grid search object with cross validation
    grid_search = GridSearchCV(svm, param_grid, cv=5, verbose=0)

    # Fit the grid search object on the training data and use cross validation for evaluation
    grid_search.fit(X_train, y_train)

    # Access the best hyperparameters and the best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Evaluate the best model on the validation set
    y_val_pred = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    # Evaluate the best model on the test set
    y_test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    class_report = classification_report(y_test, y_test_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_test_pred)

    # print(f'Best Parameters {data}:', best_params)
    # print('Validation Accuracy:', val_accuracy)
    # print('Test Accuracy:', test_accuracy, '\n')
    # print(classification_report(y_test, y_test_pred))
    # print(confusion_matrix(y_test, y_test_pred))

    info_dict = {
        'algorithm': 'SVM',
        'dataset': f'{dataset}',
        'parameter': best_params,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
        'f1-score': class_report['macro avg']['f1-score'],
        'conf_matrix': conf_matrix
    }

    return info_dict    

def KNN(dataset):
    if dataset == 'fs':
        X_train = x_train_fs
        X_val = x_val_fs
        X_test = x_test_fs
    if dataset == 'pca':
        X_train = x_train_pca
        X_val = x_val_pca
        X_test = x_test_pca   
    else:
        X_train = x_train
        X_val = x_val
        X_test = x_test

    k_range = list(range(1, 40))
    param_grid = dict(n_neighbors=k_range)

    # Create KNN classifier
    knn = KNeighborsClassifier()

    # Create a grid search object with cross validation
    grid_search = GridSearchCV(knn, param_grid, cv=5, verbose=0)

    # Fit the grid search object on the training data and use cross validation for evaluation
    grid_search.fit(X_train, y_train)

    # Access the best hyperparameters and the best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Evaluate the best model on the validation set
    y_val_pred = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    # Evaluate the best model on the test set
    y_test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    class_report = classification_report(y_test, y_test_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_test_pred)

    # print(f'Best Parameters {data}:', best_params)
    # print('Validation Accuracy:', val_accuracy)
    # print('Test Accuracy:', test_accuracy, '\n')
    # print(classification_report(y_test, y_test_pred))
    # print(confusion_matrix(y_test, y_test_pred))

    info_dict = {
        'algorithm': 'KNN',
        'dataset': f'{dataset}',
        'parameter': best_params,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
        'f1-score': class_report['macro avg']['f1-score'],
        'conf_matrix': conf_matrix
    }

    return info_dict

@ignore_warnings(category=ConvergenceWarning)
def LR(dataset):
    if dataset == 'fs':
        X_train = x_train_fs
        X_val = x_val_fs
        X_test = x_test_fs
    elif dataset == 'pca':
        X_train = x_train_pca
        X_val = x_val_pca
        X_test = x_test_pca 
    else:
        X_train = x_train
        X_val = x_val
        X_test = x_test
        
    param_grid = {
    'C': [0.1, 0.5, 1],
    'solver': ['lbfgs', 'newton-cg', 'liblinear', 'saga'],
    'max_iter': [10, 50, 100]
    }

    model = LogisticRegression()
    #model.fit(X_train,y_train)

    # Create a grid search object with cross validation (number of cross-validation folds cv=5)
    grid_search = GridSearchCV(model, param_grid, cv=5, verbose=0)

    # Fit the grid search object on the training data and use cross validation for evaluation
    grid_search.fit(X_train, y_train)

    # Access the best hyperparameters and the best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Evaluate the best model on the validation set
    y_val_pred = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    # Evaluate the best model on the test set
    y_test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    class_report = classification_report(y_test, y_test_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_test_pred)

    # print(f'Best Parameters {data}:', best_params)
    # print('Validation Accuracy:', val_accuracy)
    # print('Test Accuracy:', test_accuracy, '\n')
    # print(classification_report(y_test, y_test_pred))
    # print(confusion_matrix(y_test, y_test_pred))

    info_dict = {
        'algorithm': 'LR',
        'dataset': f'{dataset}',
        'parameter': best_params,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
        'f1-score': class_report['macro avg']['f1-score'],
        'conf_matrix': conf_matrix
    }

    return info_dict


# Predictions with best parameters to generate files and summarize
@ignore_warnings(category=ConvergenceWarning)
def Prediction():
    alg_dict = {
        'name': ['SVM', 'KNN', 'LR'],
        'dataset': ['raw', 'fs', 'pca']
    }
    
    RF_scores = []
    RF_params = []
    SVM_scores = []
    SVM_params = []
    KNN_scores = []
    KNN_params = []
    LR_scores = []
    LR_params = []

    # Run all algorithms on all datasets and score them by: (validation accuracy + test accuracy) * macro average f1-score
    for i, n in enumerate(alg_dict['dataset']):
        info_dict_RF = RF(n)
        RF_params.insert(i, info_dict_RF['parameter'])
        score = (float(info_dict_RF['val_accuracy']) + float(info_dict_RF['test_accuracy'])) * float(info_dict_RF['f1-score'])
        RF_scores.insert(i , score)

        info_dict_SVM = SVM(n)
        SVM_params.insert(i, info_dict_SVM['parameter'])
        score = (float(info_dict_SVM['val_accuracy']) + float(info_dict_SVM['test_accuracy'])) * float(info_dict_SVM['f1-score'])
        SVM_scores.insert(i , score)
        
        info_dict_KNN = KNN(n)
        KNN_params.insert(i, info_dict_KNN['parameter'])
        score = (float(info_dict_KNN['val_accuracy']) + float(info_dict_KNN['test_accuracy'])) * float(info_dict_KNN['f1-score'])
        KNN_scores.insert(i , score)
        
        info_dict_LR = LR(n)
        LR_params.insert(i, info_dict_LR['parameter'])
        score = (float(info_dict_LR['val_accuracy']) + float(info_dict_LR['test_accuracy'])) * float(info_dict_LR['f1-score'])
        LR_scores.insert(i , score)
        
        print(f'Trained all algorithms on dataset {n}')
    
    # Determine which dataset performs the best and set dataset to predict
    datasets_scores = []
    for i, n in enumerate(RF_scores):
        score = n + SVM_scores[i] + KNN_scores[i] + LR_scores[i]
        datasets_scores.insert(i, score)
    best_dataset = datasets_scores.index(max(datasets_scores))
    
    print(f'{alg_dict["dataset"][best_dataset]} was chosen to be the best performing dataset with a score of: {max(datasets_scores)}')

    # Select correct datasets for best_dataset predictions
    if best_dataset == 0:
        x_finalpred = x_train
        x_finalpredtest = x_test
        colnames_final = column_names
    elif best_dataset == 1:
        x_finalpred = x_train_fs
        x_finalpredtest = x_test_fs
        colnames_final = column_names
        for i in selected_feature_indices:
            del colnames_final[i]
    elif best_dataset == 2:
        x_finalpred = x_train_pca
        x_finalpredtest = x_test_pca
        colnames_final = column_names[:n_components]
    else:
        print(f'Error while choosing best_dataset with value {best_dataset}')

    # run all algorithms on selected dataset with best parameters and write to full column (oversampled) .csv file
    for alg in alg_dict['name']:
        if alg == 'RF':
            params = RF_params[best_dataset]
            model = RandomForestClassifier(n_estimators = int(params["n_estimators"]),
                                           max_depth = int(params["max_depth"]),
                                           min_samples_split = int(params["min_samples_split"]),
                                           random_state=42)
        elif alg == 'SVM':
            params = SVM_params[best_dataset]
            model = SVC(C = float(params["C"]),
                        gamma = float(params["gamma"]),
                        kernel = params["kernel"],
                        random_state=42)
        elif alg == 'KNN':
            params = KNN_params[best_dataset]
            model = KNeighborsClassifier(n_neighbors = int(params["n_neighbors"]))

        elif alg == 'LR':
            params = LR_params[best_dataset]
            model = LogisticRegression(C = float(params["C"]),
                                       solver = params["solver"],
                                       max_iter = int(params["max_iter"]))
            
        model.fit(x_finalpred, y_train)
        pred = model.predict(x_finalpredtest)

        # Model Summary
        print(f'\n{alg} Summary:')
        print(f'Best Parameters {params}')
        print('Validation Accuracy:', accuracy_score(y_test, pred))
        print('Test Accuracy:', accuracy_score(y_test, pred), '\n')
        print(classification_report(y_test, pred))
        print("Confusion Matrix")
        print(confusion_matrix(y_test, pred))

        DF_pred = pd.DataFrame(pred, columns = ['Predicted Transaction'])
        DF_y = pd.DataFrame(y_test, columns = ['Transaction'])
        DF_x = pd.DataFrame(x_finalpredtest, columns = colnames_final)
        DF_full = pd.concat([DF_pred, DF_y, DF_x], axis='columns')
        DF_full.to_csv(f'{alg}-{alg_dict["dataset"][best_dataset]}-prediction.csv')
    
    print('\n Predictions successfully written to files.')

def main():
    Prediction()

main()