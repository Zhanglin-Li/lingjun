"""Update Kaggle datasets: models and code."""

from janestreet.kaggle import update_dataset, upload_code
from janestreet.config import PATH_MODELS, PATH_CODE
from janestreet.setup_env import setup_environment

setup_environment(track=False)

update_dataset("janestreet2025-models", PATH_MODELS)
upload_code("janestreet2025-code", PATH_CODE)
