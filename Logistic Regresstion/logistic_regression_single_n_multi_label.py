# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
'''
CSV file which tells which of the users purchased/not purchased a particular product
'''
# Importing the dataset
dataset = pd.read_csv('C:/Users/mpandey1/Desktop/ML using Python Training/day5/Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


classifier.coef_
classifier.intercept_

# Predicting the Test set results
y_pred = classifier.predict(pd.DataFrame(X_test))

y_pred1 =classifier.predict_proba(pd.DataFrame(X_test) )
y_pred1=pd.DataFrame(y_pred1)
y_pred1.columns=['not_purchased','purchased']

#y_pred[0]=1
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix , classification_report , roc_curve,auc
cm = confusion_matrix(y_test, y_pred)

pd.crosstab(y_pred,y_test)

total1=sum(sum(cm))

#####from confusion matrix calculate accuracy
accuracy=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy)

sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity )

specificity = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity)

precision = cm[0,0]/(cm[0,0]+cm[1,0])
print('precision : ', precision )

classifier.score(X_test, y_test)

print(classification_report(y_test, y_pred))

def cutoff(x):
    x =np.where(x>0.9,0,1)
    return x
    
y_pred2=cutoff(y_pred1.not_purchased) 

cm1 = confusion_matrix(y_test, y_pred2)

accuracy1=(cm1[0,0]+cm1[1,1])/total1
print ('Accuracy : ', accuracy)

sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity : ', specificity1)

precision1 = cm1[0,0]/(cm1[0,0]+cm1[1,0])
print('precision : ', precision1 )

precision2 = cm1[1,1]/(cm1[1,1]+cm1[0,1])
print('precision : ', precision2 )


classifier.score(X_test, y_test)

print(classification_report(y_test, y_pred2))   



fpr ,tpr , _  =roc_curve(y_test, y_pred1.iloc[:,1])
    
df_roc = pd.DataFrame(dict(fpr=fpr,tpr=tpr))

AUC=auc(fpr,tpr)


plt.title("ROC Curve")
plt.plot(fpr,tpr, label='auc = %0.2f' % AUC)
plt.legend(loc="lower right")
plt.plot([0,0],[0,1],'r--')
plt.show()

##multi label
data =pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data')

data.head()
'''
   1. Sample code number            id number
   2. Clump Thickness               1 - 10
   3. Uniformity of Cell Size       1 - 10
   4. Uniformity of Cell Shape      1 - 10
   5. Marginal Adhesion             1 - 10
   6. Single Epithelial Cell Size   1 - 10
   7. Bare Nuclei                   1 - 10
   8. Bland Chromatin               1 - 10
   9. Normal Nucleoli               1 - 10
  10. Mitoses                       1 - 10
  11. Class:                        (2 for benign, 4 for malignant)
'''
data.columns=['code_num','clump_thick','cell_size','cell_shape','marginal_adhesion','single_epi','bare_nucei',
              'bland_chr','normal_nuc','mitoses','class']
data.head()

data.isnull().any()

data.isnull().values.any()

data.isnull().sum().sum()

data.count()

X1=data1.iloc[:,2:10]

Y1=data1.iloc[:,-1]

from sklearn.cross_validation import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1,Y1, test_size = 0.25, random_state = 1234)

from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(random_state = 0)
classifier1.fit(X_train1, y_train1)

data1=data.loc[data['bare_nucei']!='?']


classifier1.coef_
classifier1.intercept_


classifier1.score(X_test1,y_test1)

pr=classifier1.predict(X_test1)

print( metrics.confusion_matrix(y_test1,pr))


from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
digit=load_digits()
df=pd.DataFrame(digit)

df.head(5)

X2=digit['data']
Y2=digit['target']


X_train2, X_test2, y_train2, y_test2 = train_test_split(X2,Y2, test_size = 0.25, random_state = 0)

X_train2.shape

from sklearn.linear_model import LogisticRegression

mod=LogisticRegression(solver='newton-cg',multi_class='multinomial')

mod.fit(X_train2,y_train2)

mod.score(X_test2,y_test2)

pred_digit=mod.predict(X_test2)

print( metrics.confusion_matrix(y_test2,pred_digit))