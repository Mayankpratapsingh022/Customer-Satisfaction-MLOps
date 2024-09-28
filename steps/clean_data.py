import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessStrategy
from typing_extensions import Annotated
from typing import Tuple


@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, 'X_train'],
    Annotated[pd.DataFrame, 'X_test'],
    Annotated[pd.Series, 'y_train'],
    Annotated[pd.Series, 'y_test'],
]:
    """ Cleans the data and divides it into train and test sets.

    Args:
        df: Raw data

    Returns:
        X_train: Training data
        X_test: Testing data
        y_train: Training labels
        y_test: Testing labels
    """
    try:
        # Data Preprocessing Strategy
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        process_data = data_cleaning.handle_data()

        # Check for NaN values after preprocessing
        if process_data.isnull().sum().sum() > 0:
            logging.warning("NaN values still exist after data cleaning. Imputing remaining NaN values.")
            process_data = process_data.fillna(process_data.median())  # Fill any remaining NaNs with median

        # Data Division Strategy
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(process_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()

        # Final check for NaN values in the target (y) columns
        if y_train.isnull().sum() > 0 or y_test.isnull().sum() > 0:
            logging.warning("NaN values detected in the target column. Dropping rows with NaN values.")
            X_train = X_train[~y_train.isnull()]
            y_train = y_train[~y_train.isnull()]
            X_test = X_test[~y_test.isnull()]
            y_test = y_test[~y_test.isnull()]

        logging.info("Data cleaning completed successfully")
        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error('Error in cleaning data: {}'.format(e))
        raise e
