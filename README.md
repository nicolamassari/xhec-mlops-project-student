<div align="center">

# Abalone Age Prediction API

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)]()
[![FastAPI](https://img.shields.io/badge/fastapi-0.68.0-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-enabled-brightgreen.svg)](https://www.docker.com/)

</div>

## Project Overview

The **Abalone Age Prediction API** is a FastAPI-based application that predicts the age of an abalone based on its physical measurements and sex. The prediction is expressed as the number of rings in the abalone, which correlates with its age. This API provides endpoints to serve predictions using a machine learning model and supports robust deployments using Docker and Prefect for server management.

### Features
- **FastAPI**: A modern web framework for building APIs with Python.
- **Pydantic**: For data validation and serialization of incoming requests.
- **MLflow**: Integrated to track model experiments.
- **Prefect**: A server to orchestrate workflows.
- **Docker**: Containerization for consistent deployments.

## Team Members
- **Nicola Massari** - [GitHub Profile](https://github.com/nicolamassari)
- **Alexandra Catalina Negoita** - [GitHub Profile](https://github.com/Catalina-13)
- **Dora Bijvoet** - [GitHub Profile](https://github.com/dorabijvoet)
- **Jack Khoueiri** - [GitHub Profile](https://github.com/jackkhoueiri)
- **Yanqing Mao** - [GitHub Profile](https://github.com/yanqing-mao)


## Requirements
- Python 3.8+
- FastAPI
- Uvicorn
- Prefect
- Pydantic
- Scikit-learn
- Docker (for containerized deployment)

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository_url>
   cd project-root

2. **Set Up a Virtual Environment**
  python -m venv mlops-env
  source mlops-env/bin/activate  # For Windows: .\mlops-env\Scripts\activate

3. **Install Dependencies**
   pip install -r requirements.txt

4. **Model and Preprocessor Preparation**
   Make sure the trained model and preprocessor files are located at the paths specified in app_config.py. For instance:

    local_models/preprocessor_v0.0.1.pkl
    local_models/model_v0.0.1.pkl

## Running the Application
1. **Start the Application Locally**
Run the FastAPI app using Uvicorn:
uvicorn src.web_service.app:app --reload --host 0.0.0.0 --port 8000
Access the app at http://localhost:8000.
View API documentation at http://localhost:8000/docs.

2. **Running via Docker**
2.1. Build the Docker Image
docker build -t abalone-age-prediction-app .
2.2. Run the Docker Container
docker run -p 8001:8001 abalone-age-prediction-app
The app will be accessible at http://localhost:8001.


## Prefect Integration
The project uses Prefect to manage workflows. To start the Prefect server and run the app, use the following commands:
# Make sure the script is executable
chmod +x bin/run_services.sh

# Run Prefect and FastAPI services
./bin/run_services.sh


## Endpoints
1. **Health Check**
URL: /
Method: GET
Response: {"health_check": "App up and running!"}

2. **Prediction Endpoint**
URL: /predict
Method: POST
Description: Predicts the number of rings (indicative of age) based on input data.

Request Body:
{
  "length": 0.455,
  "diameter": 0.365,
  "height": 0.095,
  "whole_weight": 0.514,
  "shucked_weight": 0.2245,
  "viscera_weight": 0.101,
  "shell_weight": 0.15,
  "sex": "M"
}

Response:
{
  "predicted_rings": 10.5
}

## Development and Contribution
1. **Pre-Commit Hooks:** Pre-commit hooks ensure code quality. To set them up:
pre-commit install
2. **Testing:** Add tests in a dedicated tests/ folder and use pytest to run tests
   pytest

3. **Code** Formatting and Linting: The project uses tools like black and flake8 for formatting and linting:
black .
flake8 .

## Deployment
To deploy the app, you can either build and push the Docker image to a container registry or use any cloud service that supports Docker deployments.

## Troubleshooting
Port Issues: Ensure that the ports exposed in Docker and on your local machine do not conflict with other services.
Model Loading Issues: Ensure that the model and preprocessor paths in app_config.py are correct.

   



