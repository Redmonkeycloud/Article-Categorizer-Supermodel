import pickle
import pandas as pd
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Collect the data.
data = pd.read_csv('/content/training_data.csv')
data['category'].unique()
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(data['category'])
data['label'] = label_encoder.transform(data['category'])
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

# Step 2: Preprocess the data.
X = data['body']
y = data['label']

vectorized_x = vectorizer.fit_transform(X)

# Step 2: Initialize machine learning models.
nb = MultinomialNB()
svm = SVC(kernel='linear', probability=True)
rf = RandomForestClassifier()

# Step 4: Train machine learning models.
nb.fit(vectorized_x, y)
svm.fit(vectorized_x, y)
rf.fit(vectorized_x, y)

# Step 5: Combine the models with voting classifier.
voting_clf = VotingClassifier(estimators=[('nb', nb), ('svm', svm), ('rf', rf)], voting='soft')
voting_clf.fit(vectorized_x, y)

# Step 6: Save the model.
pickle.dump(voting_clf, open('financial_text_classifier.pkl', 'wb'))
pickle.dump(vectorizer, open('financial_text_vectorizer.pkl', 'wb'))
pickle.dump(label_encoder, open('financial_text_encoder.pkl', 'wb'))