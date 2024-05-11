import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

import seaborn as sns
import contractions
import pickle
import re
import string
from sklearn.utils import resample
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

seed = 4353

reddit_data1 = "input_datasets\DASH_2020_Drug_Data.csv"
reddit_data2 = "input_datasets\AskReddit.csv"
reddit_data3= "input_datasets\GradSchool_data.csv"

corpus1 = pd.read_csv(reddit_data1)
corpus2 = pd.read_csv(reddit_data2)
corpus3 = pd.read_csv(reddit_data3)

print(corpus1.head(1))
print(corpus2.head(1))
print(corpus3.head(1))

#Eliminating the uncommon columns for all 3 corpora
corpus1.drop(columns=['id', 'username', '#comments', 'score', 'label_recommendation'], inplace=True)
print(corpus1)

corpus2.drop(columns=['score', 'domain', 'id', 'ups'], inplace = True)
print(corpus2)

corpus3.drop(columns=['ID', 'Flair', 'is_Original', 'num_comments', 'Subreddit', 'URL', 'Upvotes', 'Comments', 'creation_date'], inplace = True)
print(corpus3)

for x in [corpus1, corpus2, corpus3]:
  x.dropna(inplace=True)
  x.drop_duplicates(inplace=True)

#creating a new column for the Ask Reddit data which needs labeling
corpus2['label_classification'] = 'Unrelated'
corpus3['label_classification'] = 'Unrelated'
#Keeping the same terminology: 'unrelated' for the 3rd class
corpus1['label_classification'].replace('Others', 'Unrelated', inplace = True)

corpus3.rename(columns={'Title': 'title'}, inplace = True)
corpus3.rename(columns={'Body': 'body'}, inplace = True)

# Keeping only the Recovery label no matter if it's early, medium or advanced
corpus1['label_classification'].replace('E-Recovery', 'Recovery', inplace=True)
corpus1['label_classification'].replace('M-Recovery', 'Recovery', inplace=True)
corpus1['label_classification'].replace('A-Recovery', 'Recovery', inplace=True)

concat_corpus = pd.concat([corpus1, corpus2], ignore_index=True)
print(concat_corpus)

corpus = pd.concat([concat_corpus, corpus3], ignore_index=True)
print(corpus)

counts = corpus['label_classification'].value_counts()

counts.plot(kind='bar', color='blue')
plt.xlabel('Label Classification')
plt.ylabel('Count')
plt.title('Histogram of Label Classification')
plt.show()

print(corpus['label_classification'].value_counts())
#We see that the 'Unrelated' class has more entries than 'Addicted' or 'Recovery'. This means that the data is imbalanced and downsampling will be required.

# Storing all text data(title and body) in the same entry
X_data = corpus['title'] + ' ' + corpus['body']
y_data = corpus['label_classification']
X_data = X_data.astype(str)

#Building a dataframe with the stored text data
X_data_df = pd.DataFrame(data=X_data)
X_data_df.columns = ['reddit_confessions']
X_data_df.head()

class Preprocessing:
    def __init__(self):
        self.count_vectorizer = CountVectorizer(max_features=1000)  #this converts a collection of text documents into a matrix of token counts
        self.tfidf = TfidfTransformer()  # this weighs the importance of words in a document relative to the entire corpus
        self.lemmatizer = WordNetLemmatizer()
        self.count_vectorizer.fit(X_data)
        self.tfidf.fit(self.count_vectorizer.transform(X_data))

    def __call__(self, text_data):
        return self.apply_methods(text_data)
    
    def vocab_size(self, text_data):
        all_words = " ".join(text_data).split()
        return len(set(all_words))

    def tokenization(self, text_data):
        corpus = nltk.word_tokenize(text_data)
        return corpus

    def lemmatization(self, text_data):
        corpus = self.lemmatizer.lemmatize(text_data)
        return corpus

    def delete_punctuation(self, text_data):
      corpus = text_data.lower()
      corpus_without_punct = ''.join([' ' if char in string.punctuation else char for char in corpus])
      return corpus_without_punct

    def delete_stopwords(self, text_data):
      stop_words_pattern = re.compile(r'\b(?:{})\b\s*'.format('|'.join(stopwords.words('english'))))
      cleaned_text = stop_words_pattern.sub('', text_data)
      return cleaned_text

    def apply_methods(self, text_data):

        text_data_deleted_stopwords = [self.delete_stopwords(data) for data in text_data]
        text_data_deleted_punctuation = [self.delete_punctuation(data) for data in text_data_deleted_stopwords]
        text_data_tokenized = [self.tokenization(data) for data in text_data_deleted_punctuation]
        text_data_lemmatized = [self.lemmatization(" ".join(data)) for data in text_data_tokenized]
        print('The vocabulary size is:', self.vocab_size(text_data_lemmatized))
        text_data_vector = self.count_vectorizer.fit_transform(text_data_lemmatized).toarray()
        text_data_tfidf = self.tfidf.fit_transform(text_data_vector).toarray()

        return text_data_tfidf

processor = Preprocessing()
data_X = processor(X_data)
joblib.dump(processor.count_vectorizer, 'count_vectorizer.joblib')
joblib.dump(processor.tfidf, 'tfidf_transformer.joblib')

#Splitting the data into 25% test and 75% training
X_train, X_test, y_train, y_test = train_test_split(data_X, y_data, test_size=0.25, random_state= seed)

#Downsampling of the Unrelated class and oversampling of the Addicted class using SMOTE
unrelated_count = 1527  #these counts correspond to the nr of entries in the Recovery class (middle class)
recovery_count = 1527
addicted_count = 1527
sampling_strategy = {
    'Unrelated': unrelated_count,
    'Recovery': recovery_count,
    'Addicted': addicted_count
}

# Initialize SMOTE with the specified sampling strategy
smote = SMOTE(sampling_strategy=sampling_strategy)

# Apply SMOTE to oversample the minority classes (Recovery and Addicted) in training data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Checking the counts of entries in the training data per class
print("Class distribution in training data after resampling:", Counter(y_train_resampled))

# Checking the counts of entries in the test data per class
print("Class distribution in testing data:", Counter(y_test))

#Building the Random Forest model
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators= 170, random_state= seed)
rfc.fit(X_train, y_train)
predictions = rfc.predict(X_test)

#Evaluating the Random Forest Model by creating a classification report and confusion matrix

report = classification_report(y_test, predictions)
matrix = confusion_matrix(y_test, predictions)

# Calculating the F1 score and Accuracy
rfc_f1 = round(f1_score(y_test, predictions, average='weighted'), 2)
rfc_accuracy = round((accuracy_score(y_test, predictions) * 100), 2)

print("Classification Report:\n", report)
print("Confusion Matrix:\n", matrix)
print("Accuracy: ", rfc_accuracy, "%")
print("F1 Score: ", rfc_f1)

#Building the SVM model
from sklearn.svm import SVC
from sklearn.model_selection import KFold
svc = SVC(random_state=seed)
svc.fit(X_train, y_train)

# Now we tune the hyperparameters using K-fold cross-validation
kf= KFold(n_splits=5, random_state=seed, shuffle=True)

# Hyperparametric tuning using grid search
param_grid = [{'kernel':['rbf'],
              'gamma':[1e-3, 1e-4],
              'C':[1, 10, 100, 1000]},
             {'kernel':['linear'],
             'C':[1, 10, 100, 1000]}]

grid = GridSearchCV(estimator=svc, param_grid=param_grid, scoring='accuracy', cv=kf)
grid.fit(X_train, y_train)

print('Estimator: ', grid.best_estimator_)
print('Best params : \n', grid.best_params_)
print('Training Accuracy: ', grid.best_score_)

predictions = grid.predict(X_test)

print(classification_report(y_test, predictions))

svc_f1 = round(f1_score(y_test, predictions, average='weighted'), 2)
svc_accuracy = round((accuracy_score(y_test, predictions)*100), 2)

print("Accuracy : " , svc_accuracy , "%")
print("f1_score : " , svc_f1)

# Building the Naive Bayes model
MNB = MultinomialNB()
MNB.fit(X_train, y_train)
predictions = MNB.predict(X_test)

#Evaluating the Naive Bayes model
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

MNB_f1 = round(f1_score(y_test, predictions, average='weighted'), 2)
MNB_accuracy = round((accuracy_score(y_test, predictions)*100),2)

print("Accuracy : " , MNB_accuracy , "%")
print("f1_score : " , MNB_f1)

# Creating a histogram to compare models' accuracy in order to select the best performing one
models = ['Random Forest', 'SVM', 'Naive Bayes']
accuracies = [rfc_accuracy, svc_accuracy, MNB_accuracy]
f1_scores = [rfc_f1, svc_f1, MNB_f1]

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.bar(models, accuracies, color=['pink', 'purple', 'blue'])
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy')

# Set y-axis ticks for accuracy
yticks_acc = [i for i in range(0, 101, 10)]  # From 0 to 100 with intervals of 10
yticks_acc.extend([i for i in range(85, 101, 5)])  # From 85 to 100 with intervals of 5
plt.yticks(yticks_acc)

plt.subplot(1, 2, 2)
plt.bar(models, f1_scores, color=['pink', 'purple', 'blue'])
plt.title('F1 Score Comparison')
plt.ylabel('F1 Score')

plt.tight_layout()
plt.show()

#Saving the model with the highest accuracy
joblib.dump(svc, "svm_model.joblib")

