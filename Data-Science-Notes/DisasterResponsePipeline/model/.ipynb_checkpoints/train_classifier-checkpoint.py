import sys
import pandas as pd
import numpy as np
import re
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def load_data(database_filepath):
    """
    load data from database.
    
    Input parameters:
        x: 1-D array-like of data for first group
        y: 1-D array-like of data for second group
        alternative: type of test to perform, {'two-sided', less', 'greater'}
    
    Output value:
        p: estimated p-value of test
    """

    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df[['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']]
    return X, Y, Y.columns


def tokenize(text):

    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = [re.sub(r"[^a-zA-Z0-9]", " ", lemmatizer.lemmatize(w).lower().strip()) for w in tokens]

    return clean_tokens


def build_model(tuning=False):
    # Machine Learning Pipeline with tuned Hyperparameters
    pipeline = Pipeline([
                ("vect", CountVectorizer(tokenizer=tokenize)),
                ("tfidf", TfidfTransformer()),
                ("multi_clf", MultiOutputClassifier(RandomForestClassifier(random_state = 42, 
                                                                           bootstrap = False,
                                                                           n_estimators = 200)))
    ])

    if tuning:
        # specify parameters for grid search
        n_estimators = [100,200]
        # Number of features to consider at every split
        max_features = ['auto', 'log2']
        # Maximum number of levels in tree
        max_depth = [None, 100, 200, 300, 500]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]

        parameters = {
            'multi_clf__estimator__n_estimators': n_estimators,
            'multi_clf__estimator__max_features': max_features,
            'multi_clf__estimator__max_depth': max_depth,
            'multi_clf__estimator__bootstrap': bootstrap,
        }

        # create grid search object
        cv = GridSearchCV(pipeline, param_grid = parameters, verbose=10)

        return cv
    else:
        return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    # Make predictions on X_test
    Y_pred = model.predict(X_test)
    
    for i, column in enumerate(category_names):
        report = classification_report(Y_test[column], Y_pred[:,i], output_dict=True, zero_division=0)
        # majority = (y_test[column] == (y_test[column].mode())[0]).sum()/y_test.shape[0]
        try:
            print(column, 
                  'f1-score = {:.3f}'.format(report['weighted avg']['f1-score']), 
                  # 'majority = {:.3f}'.format(majority),
                  'precision = {:.3f}'.format(report['weighted avg']['precision']),
                  'recall = {:.3f}'.format(report['weighted avg']['recall']))
        except Exception as err:
            print(column, err)


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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