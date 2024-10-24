# app_config.py

# MODELS
MODEL_VERSION = "0.0.1"
PATH_TO_PREPROCESSOR = f"local_models/preprocessor_v{MODEL_VERSION}.pkl"
PATH_TO_MODEL = f"local_models/model_v{MODEL_VERSION}.pkl"
CATEGORICAL_VARS = ["sex"] 

# MISC
APP_TITLE = "AbaloneAgePredictionApp"
APP_DESCRIPTION = "An API to predict the age of an Abalone (in terms of rings), given its physical measurements and sex."
APP_VERSION = "0.0.1"
