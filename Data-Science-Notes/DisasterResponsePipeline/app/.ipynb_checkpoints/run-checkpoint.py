import json
from numpy.lib.function_base import quantile
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify

import plotly
import plotly.graph_objects as pltgo
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = re.sub(r"[^a-zA-Z0-9]", " ", lemmatizer.lemmatize(tok).lower().strip())
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Graph1 is a Pie showing the distribution of message genres in our data
    graph1 = {
            'data': [
                pltgo.Pie(
                    labels=genre_names,
                    values=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
            }
        }

    classes = [column for column in df.columns if column not in {'id','message','original','genre'}]
    class_counts = df[df[classes] != 0][classes].count()
    
    graph2 = {
            'data': [
                pltgo.Bar(
                    x=classes,
                    y=class_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Classes',
                'yaxis': {
                    'title': "Count",
                },
                'xaxis': {
                    'title': "Class",
                    'categoryorder':"total descending"
                }
            }
        }

    df['length'] = df['message'].apply(len)
    classes.remove('child_alone')

    class_lengths_mean = [int(df[df[column] != 0]['length'].mean()) for column in classes]

    graph3 = {
            'data': [
                pltgo.Bar(
                    x=classes,
                    y=class_lengths_mean,
                    name = 'mean'
                ),             
            ],

            'layout': {
                'title': 'Distribution of Message Lengths',
                'yaxis': {
                    'title': "Message Length"
                },
                'xaxis': {
                    'title': "Class",
                    'categoryorder':"total descending",
                }
            }
        }

    graphs = [
        graph1,
        graph2,
        graph3,
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=3001, debug=True)


if __name__ == '__main__':
    main()