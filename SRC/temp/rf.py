import _pickle
import scipy.io
matr = scipy.io.loadmat('Right.mat')
matl = scipy.io.loadmat('Left.mat')
matu = scipy.io.loadmat('Up.mat')
matd = scipy.io.loadmat('Down.mat')


#print(hjorth_feature)
PSD_feature = np.ndarray(shape=(432,12))
PSD_feature = np.vstack((matl['PSD'],matr['PSD'],matu['PSD'],matd['PSD']))

l=[]
#108 labels for each direction
for i in range(0,108):
    l.append(int(0))
for i in range(0,108):
    l.append(int(1))
for i in range(0,108):
    l.append(int(2))
for i in range(0,108):
    l.append(int(3))

l=np.array(l)

from sklearn import preprocessing, cross_validation, svm
import numpy as np
from sklearn                        import metrics, svm
from sklearn.linear_model           import LinearRegression
from sklearn.linear_model           import LogisticRegression
from sklearn.tree                   import DecisionTreeClassifier
from sklearn.neighbors              import KNeighborsClassifier
from sklearn.discriminant_analysis  import LinearDiscriminantAnalysis
from sklearn.naive_bayes            import GaussianNB
from sklearn.svm                    import SVC
from sklearn import linear_model
import statistics 

X = PSD_feature

X1=preprocessing.MinMaxScaler()
# X=X1.fit_transform(X)
y=l
score=[]
print('Feature : PSD_feature')
print('4 Class problem')


# from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier
score=[]
clf=RandomForestClassifier()#100,alpha=0.0001,max_iter=1350)
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, random_state=None)
# X is the feature set and y is the target
for train_index, val_index in skf.split(X,y): 
#     print("Train:", train_index, "Validation:", val_index) 
    X_train, X_test = X[train_index], X[val_index] 
    y_train, y_test = y[train_index], y[val_index]
#    print(np.shape(hjorth_feature)) print(y_train)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    score.append(confidence)
print(score)
print('RandomForestClassifier:',statistics.mean(score))



