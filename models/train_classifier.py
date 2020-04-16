# import packages
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt', 'wordnet','stopwords', 'averaged_perceptron_tagger')
import pickle
import re
import sys
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sqlalchemy import create_engine

def load_data(data_filepath):
    # read in file
    '''
    Load the filepath and return the data
    INPUT --
        database_filepath - Filepath used for importing the database     
    OUTPUT --
        Returns the following variables:
        X - Returns the input features.  Specifically, this is returning the messages column from the dataset
        Y - Returns the categories of the dataset.  This will be used for classification based off of the input X
        y.keys - Just returning the columns of the Y columns
        
    '''
    engine = create_engine('sqlite:///data/DisasterResponse.db') ## connecting the database
    df =  pd.read_sql_table('DisasterResponse', engine)
    df = pd.read_sql_table
    print(df.head())
    x = df['message']
    y = df[df.columns[4:]]
    category_names = y.columns
    return x, y, category_names


def tokenize(text):
    '''
    this function does tokenize the text, and return a clean text instead 
    clean text -- (tokenized, lower cased, stripped, and lemmatized)
    
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls :
        text = text.replace(url, urlplaceholder)
        
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens =[]
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    '''
    trains features and labels for use by GridSearchCV
    returns a pipeline model
    
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {'clf__estimator__max_depth': [10, 50, None],
              'clf__estimator__min_samples_leaf':[2, 5, 10]}

    cv = GridSearchCV(pipeline, parameters)
    return cv


def evaluate_model(model, x_test, y_test, category_names):
    '''
    evaluate the pipeline model using x_test, y_test, category_names
    Prints the accuracy rate of each category
    
    '''
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred, target_names=category_names))
    results = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])


def save_model(model, model_filepath):
    '''
    saves the model to disk
    
    '''
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    '''
    loads the data, saves the model
    
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
        print('Building model...')
        model = build_model()
        print('Training model...')
        model.fit(x_train, y_train)
        print('Evaluating model...')
        evaluate_model(model, x_test, y_test, category_names)
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        
        print('Trained model saved!')
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()