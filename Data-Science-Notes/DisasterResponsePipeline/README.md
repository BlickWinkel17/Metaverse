# Disaster Response Pipeline Project

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Components](#project-components)
3. [File Description](#file-description)
4. [Installation](#installation)
5. [Possible Improvements](#possible-improvements)

## Project Overview
This project is part of the [Udacity Data Science Nano Degree Program](https://www.udacity.com/course/data-scientist-nanodegree--nd025). This project will analyze a [data set](https://github.com/BlickWinkel17/DisasterResponsePipeline/tree/master/data) containing real messages that were sent during disaster events. Those messages are sent from social media or from disaster response organizations. This project will build a ETL pipeline to load and process data, and a machine learning pipeline to classify those messages so as to send them to an appropriate disaster relief agency.

## Project Components
There are three components in the project.

### 1. ETL Pipeline
- Loads the message.csv and categories.csv files 
- merges two datasets
- clean data 
- stores it in a SQLite database

### 2. ML Pipeline
- Load cleaned data from database
- Build a test processing and maching learning pipline
- Use random forest classification and evaluate accuracy, precision and recall
- Train and tunes a model using GridSearchCV
- Notice that due to the size limit of a single file in Github, the output model is not a optimized one, for example, __number of estimators__ and __max_depth__ are both set small. In __train_classifier.py__, a __tuning__ parameter is also provided in the __build_model__ function, so any one can train better models locally.

### 3. Flask Web App
There is a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

## File Description
```sh
- README.md: read me file
- ETL Pipeline Preparation.ipynb: ETL pipeline preparation code
- ML Pipeline Preparation.ipynb: ML pipeline preparation code
- \app
	- run.py: flask file to run the app
   	- \templates
		- master.html: main page of the web application 
		- go.html: result web page
- \data
	- disaster_categories.csv: categories dataset
	- disaster_messages.csv: messages dataset
	- DisasterResponse.db: disaster response database
	- process_data.py: ETL process to clean up data
- \model
	- train_classifier.py: classification code
	- classifier.pkl: model saved as a pickle file
```

## Installation
### Devendencies :
   - [python (>=3.8)](https://www.python.org/downloads/)  
   - [re](https://docs.python.org/3/library/re.html)  
   - [pandas](https://pandas.pydata.org/)  
   - [numpy](https://numpy.org/)  
   - [sqlalchemy](https://www.sqlalchemy.org/)  
   - [nltk](https://www.nltk.org/)  
   - [sys](https://docs.python.org/3/library/sys.html)  
   - [plotly](https://plotly.com/python/)  
   - [sklearn](https://sklearn.org/)  
   - [joblib](https://joblib.readthedocs.io/en/latest/)  
   - [flask](https://flask.palletsprojects.com/en/2.0.x/)  
   
 ### Download and Installation
 ```sh
git clone https://github.com/BlickWinkel17/DisasterResponsePipeline.git
```
 
(Optional, since DisasterResponse.db already exists) ETL pipeline that cleans and stores data in database.
 ```sh
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

(Optional, since classifier.pkl already exists)ML pipeline that trains the classifier and save it.
```sh
python model/train_classifier.py data/DisasterResponse.db model/classifier.pkl
```
Next, run the Python file run.py.
```sh
python app/run.py
```
Finally, go to http://127.0.0.1:3001/ in your web-browser.
Type a message input box and click on the Classify Message button to see the various categories that your message falls into.

## Possible Improvements

### Precison and Recall
As can be shown by the current model, precision and recall are generally over 80% and mostly more than 90%. However, since our data is heavily __balanced__ (for some categories there are few or even none different labels, as a result the prediction is simply a majority class), this result might sound more promising than it is actually.  Considering that for disaster response organizations, recall can be more important than precision, some methods to deal with unbalanced data and improve recall remain to be discussed.

#### Screenshot of the web app
![Alt text](https://github.com/BlickWinkel17/DisasterResponsePipeline/blob/master/img/message%20classification.png)
![Alt text](https://github.com/BlickWinkel17/DisasterResponsePipeline/blob/master/img/overall%20view.png)
