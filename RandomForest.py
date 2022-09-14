# Importing Libraries 
import numpy as np   
import pandas as pd  
import re  
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
  
# Import dataset 
dataset = pd.read_csv('C:/Users/Namrata/Desktop/NLP_material/Project_material/english/agr_en_train.csv', delimiter = ',')



corpus=[]  
# 1000 (reviews) rows to clean 
for i in range(0, 12500):  
      
    # column : "Review", row ith 
    review = re.sub('[^a-zA-Z]', ' ', dataset['Text'][i])  
      
    # convert all cases to lower cases 
    review = review.lower()  
      
    # split to array(default delimiter is " ") 
    review = review.split()  
      
    # creating PorterStemmer object to 
    # take main stem of each word 
    ps = PorterStemmer()  
      
    # loop for stemming each word 
    # in string array at ith row     
    review = [ps.stem(word) for word in review 
                if not word in set(stopwords.words('english'))]  
                  
    # rejoin all string array elements 
    # to create back into a string 
    review = ' '.join(review)   
      
    # append each string to create 
    # array of clean text  
    corpus.append(review)

print(corpus)

from sklearn.feature_extraction.text import CountVectorizer
# To extract max 1500 feature. 
# "max_features" is attribute to 
# experiment with to get better results 
cv = CountVectorizer()
# X contains corpus (dependent variable) 
X_train = cv.fit_transform(corpus).toarray()  
  
# y contains answers if review 
# is positive or negative 
y_train = dataset.iloc[:, 1].values 


# Fitting Random Forest Classification 
# to the Training set 
from sklearn.ensemble import RandomForestClassifier
# n_estimators can be said as number of 
# trees, experiment with n_estimators 
# to get better results  
model = RandomForestClassifier(n_estimators =100, oob_score=True, n_jobs=-1, random_state=50, max_features="auto", min_samples_leaf=50) 
                              
model.fit(X_train,y_train)




testset = pd.read_csv('C:/Users/Namrata/Desktop/NLP_material/Project_material/english/agr_en_dev.csv', delimiter = ',')
test_doc=[]
for i in range(0, 3001):  
      
    # column : "Review", row ith 
    review = re.sub('[^a-zA-Z]', ' ', testset['Text'][i])  
      
    # convert all cases to lower cases 
    review = review.lower()  
      
    # split to array(default delimiter is " ") 
    review = review.split()  
      
    # creating PorterStemmer object to 
    # take main stem of each word 
    ps = PorterStemmer()  
      
    # loop for stemming each word 
    # in string array at ith row     
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]  
                  
    # rejoin all string array elements 
    # to create back into a string 
    review = ' '.join(review)   
      
    # append each string to create 
    # array of clean text  
    test_doc.append(review)

x_test = cv.fit_transform(test_doc).toarray()
y_test = testset.iloc[:, 1].values
y_pred = model.predict(x_test)
#print(y_pred)
import pylab as plt
from sklearn.metrics import confusion_matrix
labels = ['OAG','NAG','CAG']
cm = confusion_matrix(y_test, y_pred,labels)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
