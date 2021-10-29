import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    load messages and categories from filepaths, 
    merge the data and return the DataFrame.
    
    Input parameters:
        messages_filepath: messages_filepath
        categories_filepath: categories_filepath
    
    Output value:
        df: DataFrame consisting messages and categories
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on='id')
    return df

def clean_data(df):
    """
    clean data:
    rename columns, create dummies for categories and drop duplicates.
    
    Input parameters:
        df: DataFrame consisting messages and categories
    
    Output value:
        df: cleaned dataframe
    """    
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    # select the first row of the categories dataframe
    row = categories[:1]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x.str[:-2]).values.tolist()[0]
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from `df`
    df = df.drop('categories',axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df = df[~df.duplicated(subset=['id'])]
    return df
    
    
def save_data(df, database_filename):
    """
    output dataframe to a SQLlite database located in database_filename.
    
    Input parameters:
        df: DataFrame to be outputed
        database_filename: SQLlite database location
    
    Output value:
        None
    """    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql(database_filename.split('/')[1], engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()