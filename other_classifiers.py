import re

from classifier_base import BaseClassifier

class emptyClassifier(BaseClassifier):

    def train(self):
        """
        There  is no training for a regular expression filter, but the filter could be reset to something else with each training step
        :param reset_filter:
        :param filter:
        :return:
        """
        pass

    def update_field(self):
        pass

    def predict(self, some_data=""):
        #print('Predicting {} data points using <{}> model'.format(len(self.preprocessed),self.model_name))
        self.predictions=[]
        for w in self.preprocessed:
            self.predictions.append(0)#classify all as the same, so no re-ordering happens as all are equal

        return self.predictions


class regexClassifier(BaseClassifier):



    def train(self, filter=r'(\bai\b)|(artificial intelligence)|(machine[\s-]?learn(ing)?)'):
        """
        There  is no training for a regular expression filter, but the filter could be reset to something else with each training step
        :param reset_filter:
        :param filter:
        :return:
        """
        self.filter = filter

    def update_field(self):
        pass

    def predict(self, some_data=""):
        #print('Predicting {} data points using <{}> model'.format(len(self.preprocessed),self.model_name))
        self.predictions = []
        for w in self.preprocessed:
            if re.search(self.filter,w):
                self.predictions.append(1)
            else:
                self.predictions.append(0)
        return self.predictions

####################################################DIRTY CODE
class MLClassifiers(BaseClassifier):
    #Quick and dirty, 2 architectures in one. Predict function also does evaluation#

    def predict(self):
        import pandas as pd
        import numpy as np
        from nltk.tokenize import word_tokenize
        from nltk import pos_tag
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from sklearn.preprocessing import LabelEncoder
        from collections import defaultdict
        from nltk.corpus import wordnet as wn
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn import model_selection, naive_bayes, svm
        from sklearn.metrics import accuracy_score

        Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(self.preprocessed, self.labels,
                                                                            test_size=0.3,random_state=48)
        Encoder = LabelEncoder()
        Train_Y = Encoder.fit_transform(Train_Y)
        Test_Y = Encoder.fit_transform(Test_Y)

        Tfidf_vect = TfidfVectorizer(max_features=5000)
        Tfidf_vect.fit(self.preprocessed)
        Train_X_Tfidf = Tfidf_vect.transform(Train_X)
        Test_X_Tfidf = Tfidf_vect.transform(Test_X)
        #print(Tfidf_vect.vocabulary_)

        # fit the training dataset on the NB classifier
        Naive = naive_bayes.MultinomialNB()
        Naive.fit(Train_X_Tfidf, Train_Y)
        # predict the labels on validation dataset
        predictions_NB = Naive.predict(Test_X_Tfidf)

        print("Naive Bayes Accuracy Score -> ", accuracy_score(predictions_NB, Test_Y) * 100)
        clf = classification_report(Test_Y, predictions_NB)
        print("Classification Report:\n{}".format(clf))


        SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
        SVM.fit(Train_X_Tfidf, Train_Y)
        # predict the labels on validation dataset
        predictions_SVM = SVM.predict(Test_X_Tfidf)
        # Use accuracy_score function to get the accuracy
        print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y) * 100)
        clf = classification_report(Test_Y, predictions_SVM)
        print("Classification Report:\n{}".format(clf))
