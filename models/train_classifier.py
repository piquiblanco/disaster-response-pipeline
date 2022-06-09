import pickle
import sqlite3
import sys
from datetime import datetime as dt

import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from sqlalchemy import create_engine


def load_data(database_filepath):
    """
    Functions loads data from the specified database and cleans them before returning endogenous and exogenous variables separately.
    """
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql("messages", engine)
    X = df["message"]
    y = df.drop(columns=["message", "id", "original", "genre"])
    infrequent_labels = y.sum()[y.sum() < 10].index.tolist()
    y.drop(columns=infrequent_labels, inplace=True)
    y[y > 1] = 1
    return (X, y, y.columns)


def tokenize(text):
    """
    Function tokenizes text in preparation for tf-idf embedding. Takes in text and returns clean text
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Function creates a pipeline of tf-idf and classification; returns a model ready to train.
    """
    model = Pipeline(
        [
            (
                "tf-idf vectorization",
                TfidfVectorizer(tokenizer=tokenize, ngram_range=(1, 2)),
            ),
            (
                "classifier",
                MultiOutputClassifier(
                    estimator=RandomForestClassifier(
                        class_weight="balanced", n_estimators=200, max_depth=3,
                    ),
                ),
            ),
        ],
        verbose=True,
    )
    return model


def evaluate_model(model, X, y, database_filepath):
    """
    Function evaluates model with accuracy metric; provides counts from the confusion matrix.
    """
    predictions = model.predict(X)
    predictions = pd.DataFrame(predictions, columns=y.columns)

    reports = []
    for label in list(y):
        ylab = y[label]
        predlab = predictions[label]
        rep = dict()
        rep["feature"] = label
        rep["f1"] = f1_score(ylab, predlab, average="micro")
        rep["precision"] = precision_score(
            ylab, predlab, zero_division=0, average="micro"
        )
        rep["recall"] = recall_score(ylab, predlab, average="micro")
        rep["accuracy"] = accuracy_score(ylab, predlab)

        # Confusion matrix:
        conf = pd.Series(
            confusion_matrix(ylab, predlab).ravel(), index=["tn", "fp", "fn", "tp"]
        )
        conf_dict = {ind: conf[ind] for ind in conf.index}
        rep.update(conf_dict)
        reports.append(pd.Series(rep))

    df_report = np.round(pd.DataFrame(reports).set_index("feature"), 2)
    engine = create_engine(f"sqlite:///{database_filepath}")
    df_report.to_sql("model_evaluation", engine, index=False, if_exists="replace")
    return df_report


def save_model(model, model_filepath):
    """Perform pickle dump of the model"""
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, database_filepath)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
