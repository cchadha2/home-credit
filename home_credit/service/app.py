"""View functions for home-credit service."""
import joblib
import logging
import os

# Initialize the service logger.
LOGGER = logging.getLogger(__name__)

# Initialize model.
MODEL = joblib.load(os.environ.get("MODEL_PATH"))


def health():
    """Server health check.

    Returns:
        dict: Healthy response object
    """
    return {"status": "Application is running"}


def predict(body):
    """Execute prediction workflow and return results.

    Args:
        body (dict): Request body.

    Returns:
        dict: Response object with predictions in 'results' key.
    """
    return {"output": MODEL.predict(body["input"]).tolist()}
