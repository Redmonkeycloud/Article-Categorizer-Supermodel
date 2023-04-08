import pandas as pd
import pickle

model = pickle.load(open('financial_text_classifier.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('financial_text_vectorizer.pkl', 'rb'))
label_encoder = pickle.load(open('financial_text_encoder.pkl', 'rb'))

def process_data():
  # read input file
  input = pd.read_csv('training_data.csv')
  # vectorize the data
  features = tfidf_vectorizer.transform(input['body'])
  # predict the classes
  predictions = model.predict(features)
  # convert output labels to categories
  input['category'] = label_encoder.inverse_transform(predictions)
  # save results to csv
  output = input[['id', 'category']]
  output.to_csv('results.csv', index=False)
  print(output)

process_data()