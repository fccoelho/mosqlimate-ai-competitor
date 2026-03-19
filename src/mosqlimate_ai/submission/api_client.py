"""API client for Mosqlimate platform.

Handles model registration and prediction submission to the
Mosqlimate API for the Sprint 2025 competition.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

API_BASE_URL = "https://api.mosqlimate.org/api"


class MosqlimateClient:
    """Client for interacting with Mosqlimate API.

    Handles model registration, prediction submission, and
    data retrieval for the Sprint 2025 competition.

    Args:
        api_key: API key from Mosqlimate profile (or set MOSQLIMATE_API_KEY env var)
        base_url: API base URL

    Example:
        >>> client = MosqlimateClient()
        >>> model_id = client.register_model(model_info)
        >>> client.submit_prediction(model_id, prediction_data)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = API_BASE_URL,
    ):
        self.api_key = api_key or os.getenv("MOSQLIMATE_API_KEY")
        self.base_url = base_url.rstrip("/")

        if not self.api_key:
            logger.warning(
                "No API key provided. Set MOSQLIMATE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "X-API-Key": self.api_key,
            }
        )

    def register_model(
        self,
        name: str,
        description: str,
        repository: str,
        implementation_language: str = "Python",
        disease: str = "dengue",
        temporal: bool = True,
        spatial: bool = False,
        categorical: bool = False,
        adm_level: int = 1,
        time_resolution: str = "week",
        sprint: bool = True,
    ) -> int:
        """Register a new model with Mosqlimate.

        Args:
            name: Model name
            description: Model description
            repository: GitHub repository URL
            implementation_language: Programming language
            disease: Disease being forecasted
            temporal: Whether model uses temporal features
            spatial: Whether model uses spatial features
            categorical: Whether model outputs categorical predictions
            adm_level: Administrative level (1=state, 2=country)
            time_resolution: Time resolution ('week', 'month')
            sprint: Whether this is for the Sprint

        Returns:
            Registered model ID
        """
        payload = {
            "name": name,
            "description": description,
            "repository": repository,
            "implementation_language": implementation_language,
            "disease": disease,
            "temporal": temporal,
            "spatial": spatial,
            "categorical": categorical,
            "adm_level": adm_level,
            "time_resolution": time_resolution,
            "sprint": sprint,
        }

        url = f"{self.base_url}/registry/models/"

        response = self.session.post(url, json=payload)

        if response.status_code == 201:
            model_id = response.json().get("id")
            logger.info(f"Model registered successfully with ID: {model_id}")
            return model_id
        else:
            raise RuntimeError(
                f"Failed to register model: {response.status_code} - {response.text}"
            )

    def submit_prediction(
        self,
        model_id: int,
        description: str,
        commit: str,
        predict_date: str,
        adm_0: str,
        adm_1: Optional[str],
        prediction: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Submit a prediction to Mosqlimate.

        Args:
            model_id: Registered model ID
            description: Prediction description
            commit: Git commit hash
            predict_date: Date prediction was made
            adm_0: Country code (e.g., "BRA")
            adm_1: State code (e.g., "SP") or None for national
            prediction: Prediction data dictionary

        Returns:
            API response
        """
        payload = {
            "model": model_id,
            "description": description,
            "commit": commit,
            "predict_date": predict_date,
            "adm_0": adm_0,
            "adm_1": adm_1,
            "adm_2": None,
            "adm_3": None,
            "prediction": prediction,
        }

        url = f"{self.base_url}/registry/predictions/"

        response = self.session.post(url, json=payload)

        if response.status_code == 201:
            logger.info(f"Prediction submitted successfully for {adm_1 or adm_0}")
            return response.json()
        else:
            raise RuntimeError(
                f"Failed to submit prediction: {response.status_code} - {response.text}"
            )

    def submit_all_predictions(
        self,
        model_id: int,
        submissions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Submit multiple predictions.

        Args:
            model_id: Registered model ID
            submissions: List of submission dictionaries

        Returns:
            List of API responses
        """
        responses = []

        for submission in submissions:
            response = self.submit_prediction(
                model_id=model_id,
                description=submission["description"],
                commit=submission["commit"],
                predict_date=submission["predict_date"],
                adm_0=submission["adm_0"],
                adm_1=submission["adm_1"],
                prediction=submission["prediction"],
            )
            responses.append(response)

        logger.info(f"Submitted {len(responses)} predictions")
        return responses

    def get_model(self, model_id: int) -> Dict[str, Any]:
        """Get model information.

        Args:
            model_id: Model ID

        Returns:
            Model information
        """
        url = f"{self.base_url}/registry/models/{model_id}/"
        response = self.session.get(url)

        if response.status_code == 200:
            return response.json()
        else:
            raise RuntimeError(f"Failed to get model: {response.status_code}")

    def list_models(self) -> List[Dict[str, Any]]:
        """List all models registered by the user.

        Returns:
            List of model dictionaries
        """
        url = f"{self.base_url}/registry/models/"
        response = self.session.get(url)

        if response.status_code == 200:
            return response.json().get("results", [])
        else:
            raise RuntimeError(f"Failed to list models: {response.status_code}")

    def get_predictions(
        self,
        model_id: Optional[int] = None,
        adm_1: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get predictions.

        Args:
            model_id: Filter by model ID
            adm_1: Filter by state

        Returns:
            List of predictions
        """
        url = f"{self.base_url}/registry/predictions/"
        params = {}

        if model_id:
            params["model"] = model_id
        if adm_1:
            params["adm_1"] = adm_1

        response = self.session.get(url, params=params)

        if response.status_code == 200:
            return response.json().get("results", [])
        else:
            raise RuntimeError(f"Failed to get predictions: {response.status_code}")

    def delete_prediction(self, prediction_id: int) -> bool:
        """Delete a prediction.

        Args:
            prediction_id: Prediction ID to delete

        Returns:
            True if successful
        """
        url = f"{self.base_url}/registry/predictions/{prediction_id}/"
        response = self.session.delete(url)

        if response.status_code == 204:
            logger.info(f"Prediction {prediction_id} deleted")
            return True
        else:
            raise RuntimeError(f"Failed to delete prediction: {response.status_code}")

    def test_connection(self) -> bool:
        """Test API connection.

        Returns:
            True if connection successful
        """
        try:
            response = self.session.get(f"{self.base_url}/datastore/dengue/")
            return response.status_code in [200, 401]
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


def get_git_commit_hash() -> str:
    """Get the current git commit hash.

    Returns:
        Git commit hash string
    """
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception as e:
        logger.warning(f"Could not get git commit hash: {e}")
        return "unknown"


def submit_forecasts(
    model_id: int,
    submissions: List[Dict[str, Any]],
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convenience function to submit forecasts.

    Args:
        model_id: Registered model ID
        submissions: List of submission dictionaries
        api_key: API key (optional)

    Returns:
        List of API responses
    """
    client = MosqlimateClient(api_key=api_key)
    return client.submit_all_predictions(model_id, submissions)
