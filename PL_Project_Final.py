
# coding: utf-8

# In[100]:


## course: CS571
## date: 12/5/2017
## Authors : Siddharth Kulshrestha, Sahil Kolwankar, Alankrit Jain, Pooja Upadhyay, Sachin Rodge
## Project: SMS Spam Predictor using Machine Learning


# In[102]:


#Importing Required Packages
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import wordcloud
from collections import Counter
import numpy as np
import re
import matplotlib.pyplot as plt
import os
from nltk.stem import PorterStemmer

get_ipython().magic(u'matplotlib inline')


# In[84]:


plt.figure(figsize=(30,20))
pd.set_option('display.max_colwidth', 30)


# In[86]:


def read_and_display(filename):
    df = pd.read_csv(filename,sep = "\t")
    print "Data Read Successfully \n"
    print df.head(10)


# In[87]:


read_and_display("SMSSpamCollection")


# In[11]:


#Executing the Haskell DataClearning program
os.system('ghc Data_Cleanse_PL.hs')


# In[88]:


read_and_display("output.csv")


# In[94]:


def stemming_text_file(filename):
    '''
    This function is used to Apply a Stemming method i.e Porter Stemmer on the text file.
    Stemming is a process to normalize similiar, slightly different words into same words.
    '''
    with open(filename, 'r+') as f:
        for line in f:
            singles = []

            stemmer = PorterStemmer() 
            for plural in line.split():
                singles.append(stemmer.stem(plural))
        f.seek(0)
        print ' '.join(singles)
        f.write(' '.join(singles))


# In[89]:


def print_wordcloud(column_name):
    '''
    Returns a Wordcloud for Exploratory Dataset
    '''
    wordcloud = wordcloud.WordCloud().generate(' '.join(df['text']))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    


# In[19]:


class Spam_Filter_Model(object):
    
    text_test, text_train, class_test, class_train, data_matrix, classifier = (None, None, None, None, None, None)
    
    def train_test_split(self, test_split_percent):
        '''
        This method takes a argument : test_split_percent and splits the dataset into training 
        and testing according to the specified percentage
        '''
        
        assert test_split_percent <= 0.25, "Please reduce the size of the training set \n"
        
        self.text_train, self.text_test, self.class_train, self.class_test =         train_test_split(self.dataset.text, self.dataset["class"], test_size = test_split_percent)
        
        
    def __init__(self, algorithm, dataset, vectorize, test_split_percent):
        
        self.algorithm = algorithm
        self.dataset = pd.read_csv(dataset, sep = "\t")
        self.vectorize = vectorize
        self.train_test_split(test_split_percent)
               
    def show_data(self):
        """
        Return the first five rows of the dataset
        """
        print self.dataset.head(5)
        
    def show_shape(self):
        """
        Prints out the number of Examples and number of classes in the dataset
        """
        print "The shape of the dataset is \n"
        print "Number of Examples - ", str(self.dataset.shape[0]), "\n"
        print "Number of Classes - ", str(self.dataset.shape[1]), "\n"
                
    def describe_data(self):
        """
        Returns the description of the dataset and its important attributes
        """
        return self.dataset.groupby('class').describe()
                
            
    def word_count_visualize(self):
        '''
        Returns a histogram with the size of each datapoint
        '''
        word_count_list = []
        counter = Counter(self.dataset["text"])
        for index, row in self.dataset.iterrows():
            word_count_list.append(len(re.findall(r'\w+', row["text"])))
        hist = plt.hist(word_count_list, alpha = 0.5)
        plt.show()
        
    def vectorize_train_test(self):
        '''
        This method vectorizes and trains the the training set using the specified Machine Learning
        Algorithm and tests the result with the training set
        '''
        count_vectorizer = CountVectorizer()
        counts = count_vectorizer.fit_transform(self.text_train)
        count1 = count_vectorizer.transform(self.text_test)
        
        tfidf = self.vectorize()
        tf_mat = tfidf.fit_transform(counts)
        tf_mat_test = tfidf.transform(count1)
        
        classifier = self.algorithm()

        targets = self.class_train
        classifier.fit(tf_mat, targets)
        self.classifier = classifier
        
        preds = classifier.predict(tf_mat_test)
        print 'Accuracy of the test set is: ', accuracy_score(self.class_test, preds) * 100 ,"%" "\n \n"
        
        return preds
        

    def metrics(self, preds):
        '''
        Returns essential metrics to study the classification results and help user choose the better algorithm
        '''
        print "The confusion Matrix is \n" 
        cm = confusion_matrix(self.class_test, preds)
        print cm, "\n"
        print "The classification report is: \n"
        print classification_report(self.class_test, preds)
        print "\n"
        
        TP = cm[0][0]
        FP = cm[0][1]
        FN = cm[1][0]
        TN = cm[1][1]

        specificity = (float(TN)/float(TN+FP))
        print "The specificity is ", str(specificity * 100), "%"
        


# In[22]:


model1.describe_data()


# In[95]:


model1 = Spam_Filter_Model(LogisticRegression, "SMSSpamCollection", TfidfTransformer, 0.2)
log_reg = model1.vectorize_train_test()

model1.metrics(log_reg)


# In[96]:


model2 = Spam_Filter_Model(RandomForestClassifier, "SMSSpamCollection", TfidfTransformer, 0.2)
random_forest_model = model2.vectorize_train_test()
model2.metrics(random_forest_model)


# In[151]:


model3 = Spam_Filter_Model(ExtraTreesClassifier, "SMSSpamCollection", TfidfTransformer, 0.2)
extra_tree_model = model3.vectorize_train_test()
model3.metrics(extra_tree_model)


# In[12]:


from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def index():
  return render_template('template.html')

@app.route('/my-link/')
def my_link():
  print 'I got clicked!'

  return 'Click.'

if __name__ == '__main__':
  app.run(debug=True)

