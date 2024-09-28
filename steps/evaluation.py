import logging
import pandas as pd
import mlflow
from zenml import step
from src.evaluation import MSE, RMSE, R2
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from typing import Tuple
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker



# Configure logging
logging.basicConfig(level=logging.ERROR)

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Tuple[
    Annotated[float, 'r2_score'],
    Annotated[float, 'rmse']
]:
    """
    Evaluates the model on the ingested data.
    
    Args:
       model: The regression model to be evaluated.
       X_test: The test features.
       y_test: The actual test labels.
       
    Returns:
       r2_score: The R-squared score.
       rmse: The root mean squared error.
    """
    try:
        # Make predictions
        prediction = model.predict(X_test)

        # Calculate MSE
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("mse",mse)


        # Calculate R2 score
        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("r2",r2)
        # Calculate RMSE
        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric('rmse',rmse)


        # Return R2 score and RMSE
        return r2, rmse

    except Exception as e:
        logging.error('Error in evaluating model: {}'.format(e))
        raise e
