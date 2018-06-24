import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import csv
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import sys




#Loading data
train_dataset = pd.read_csv(sys.argv[1])
validation_dataset = pd.read_csv(sys.argv[3])



#Dataset information
def info_about_dataset(df):

    print(df.head())
    print(df.describe())
    print(df.dtypes)

#info_about_dataset(train_dataset)
#info_about_dataset(validation_dataset)

columns_name = train_dataset.columns
target = train_dataset['class'].unique()



#Discard na values
def Fill_na(df):

    for i in df.columns:
        df[i] = pd.to_numeric(df[i],errors='coerce')

    for i in df.columns:
        if df[i].isnull().sum()!=0:
            df[i] = df[i].astype(np.float64)
            df[i] = df[i].fillna(df[i].mean())

    return df



#Fill na
train = Fill_na(train_dataset)
val = Fill_na(validation_dataset)


#Data shuffling
train = shuffle(train)
val = shuffle(val)


#See the range of values for each feature
'''
list1 = []
list2 = []
for i in columns_name:
    str1 = i+' '+str(val[i].max())+' - '+ str(val[i].min())
    str2 = i+' '+str(train[i].max())+' - '+ str(train[i].min())
    list1.append(str1)
    list2.append(str2)
print(list1)
print(list2)
'''


columns = train.drop(['class','objid','ra','dec','colc','rowc'],axis=1).columns
X_train = train.drop(['class','objid','ra','dec','colc','rowc'],axis=1).values
y_train = train.iloc[:,-1].values
X_val = val.drop(['class','objid','ra','dec','colc','rowc'],axis=1).values
y_val = val.iloc[:,-1].values


#Normalization
X_train = normalize(X_train,axis=0)
X_val = normalize(X_val,axis=0)



#See the range of values for each feature
'''train_ready = pd.DataFrame(X_train,columns=columns.values)
val_ready = pd.DataFrame(X_val,columns=columns.values)
list3 = []
list4 = []

for i in columns:
    str5 = i+' '+str(train_ready[i].max())+' - '+ str(train_ready[i].min())
    str6 = i+' '+str(val_ready[i].max())+' - '+ str(val_ready[i].min())
    list3.append(str5)
    list4.append(str6)
print(list3)
print(list4)
'''



def Classification_with_KNN_CV(X_train,y_train,X_val):

    knn = KNeighborsClassifier()

    k_parameters= {'n_neighbors': np.arange(1,5)}

    knn_cv = GridSearchCV(knn,param_grid=k_parameters,cv=3)

    knn_cv.fit(X_train,y_train)

    y_predict = knn_cv.predict(X_val)

    print(knn_cv.best_params_)
    print(knn_cv.best_score_)

    return y_predict



def RandomForest():
    clf = RandomForestClassifier(criterion='gini',n_estimators=84, max_depth=None, min_samples_split=2)
    #num_est = list(map(int, np.linspace(1, 100, 20)))
    '''
    bg_clf_cv_mean = []
    bg_clf_cv_std = []
    for n_est in num_est:
        bg_clf = RandomForestClassifier(n_estimators=n_est, max_depth=None, min_samples_split=2)
        scores = cross_val_score(bg_clf, X_train, y_train, cv=3, scoring='accuracy')
        bg_clf_cv_mean.append(scores.mean())
        bg_clf_cv_std.append(scores.std())
    print(bg_clf_cv_mean)
    '''
    clf.fit(X_train,y_train)
    #print(num_est)
    y_predict = clf.predict(X_val)
    return y_predict



def csv_writer(data,path):

    with open(path,'w',newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['objid','prediction_class'])
        writer.writerows(data)




def Neural_network(X_train, y_train, X_val, y_val):

    classifier = MLPClassifier(hidden_layer_sizes=(100, 100, 50), solver='adam')
    classifier.fit(X_train,y_train)
    print("Training set score: %f" % classifier.score(X_train, y_train))
    print("Test set score: %f" % classifier.score(X_val, y_val))
    y_predict = classifier.predict(X_val)
    return y_predict



def Metrix(y_val,y_predict):

    #print(classification_report(y_val,y_predict))
    print("f1_score: %f" % f1_score(y_val,y_predict,average='macro'))
    print("Cross-tabulation")
    print(pd.crosstab(y_val, y_predict, rownames=['True'], colnames=['Predicted'], margins=True))


#y_predict = Classification_with_KNN_CV(X_train,y_train,X_val)
#y_predict = RandomForest()
y_predict = Neural_network(X_train, y_train, X_val, y_val)
Metrix(y_val,y_predict)



objid = val['objid'].values
path_result = sys.argv[4]
data_result = zip(objid,y_predict)
csv_writer(data=data_result, path=path_result)



