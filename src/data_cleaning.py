import logging
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """Abstract class defining strategy for handling data."""
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):
    """Strategy for preprocessing data."""
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data."""
        try:
            # Drop unnecessary columns
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1
            )

            # Fill NaN values for specific columns
            data['product_weight_g'].fillna(data['product_weight_g'].median(), inplace=True)
            data['product_length_cm'].fillna(data['product_length_cm'].median(), inplace=True)
            data['product_height_cm'].fillna(data['product_height_cm'].median(), inplace=True)
            data['product_width_cm'].fillna(data['product_width_cm'].median(), inplace=True)
            data['price'].fillna(data['price'].median(), inplace=True)
            data['freight_value'].fillna(data['freight_value'].median(), inplace=True)
            data['product_photos_qty'].fillna(data['product_photos_qty'].median(), inplace=True)
            data['review_comment_message'].fillna('No review', inplace=True)

            # Select only numerical columns and drop the rest
            data = data.select_dtypes(include=[np.number])

            # Drop columns that are not relevant for model training
            cols_to_drop = ['customer_zip_code_prefix', 'order_item_id']
            data = data.drop(cols_to_drop, axis=1)

            return data
        except Exception as e:
            logging.error('Error in preprocessing data: {}'.format(e))
            raise e

class DataDivideStrategy(DataStrategy):
    """Strategy for dividing data into train and test sets."""
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """Divide data into train and test."""
        try:
            # Ensure no NaN values in target column
            data = data.dropna(subset=['review_score'])

            X = data.drop(['review_score'], axis=1)
            y = data['review_score']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error('Error in dividing data: {}'.format(e))
            raise e

class DataCleaning:
    """Class for cleaning data which preprocesses the data and divides it into train and test sets."""
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data."""
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load data and handle missing file error
    try:
        data = pd.read_csv('data/olist_customers_dataset.csv')
    except FileNotFoundError:
        logging.error("CSV file not found at the specified path.")
        raise

    # Preprocess and handle the data
    data_cleaning = DataCleaning(data, DataPreProcessStrategy())
    cleaned_data = data_cleaning.handle_data()

    # Check for any remaining NaN values
    if cleaned_data.isnull().sum().sum() > 0:
        logging.warning("There are still NaN values present in the data after cleaning.")
    else:
        logging.info("Data cleaning successful. No NaN values present.")
