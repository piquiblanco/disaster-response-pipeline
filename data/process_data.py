# import libraries
import sys

import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Function loads datasets according to specified paths and returns a merged dataset.
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories)
    return df


def clean_data(df):
    """
    The function removes duplicates and converts exogenous variables to binary.
    """
    categories = df.categories.str.split(";", expand=True)
    row = categories.loc[0]
    category_colnames = [x[:-2] for x in row.tolist()]
    categories.columns = category_colnames
    for column in categories:
        print(column)
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype("int")
        categories.loc[categories[column] > 1, column] = 1
    df.drop(columns="categories", inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    # df[df > 1] = 1  # convert to binary
    return df


def save_data(df, database_filename):
    """
    The function saves data in a database which path is specified in the function argument.
    """
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql("messages", engine, index=False, if_exists="replace")
    print(pd.read_sql("messages", engine).head())


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()
