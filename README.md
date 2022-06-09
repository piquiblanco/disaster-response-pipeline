# Disaster Response Pipeline Project
Project contains a Flask App which you can use to categorize different messages depending on their topic within area of disaster responses. The repository also contains a dataset of training data, as well as pipeline with classification models used to predict labels for new messages.

### File description
Project consists of following Python scripts:
- `data/process_data.py`: reads data from .csv files, cleans it and loads it into a SQLite database;
- `models/train_classifier.py`: trains a ML pipeline on available data and saves the trained model as a pickle file.
- `app/run.py`: runs the Flask app which you can use in the web browser.

### Instalation
Conda environment containing all required packages is placed in the `environment.yml` file.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/
