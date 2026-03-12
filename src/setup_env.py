"""Set up project environment."""

import os


def setup_environment(track: bool = False):
    """Set up project environment.

    Args:
        track (bool, optional): Whether to set up Weights & Biases. Defaults to True.
    """
    # Dotenv
    print("Loading environment variables from .env file...")
    from dotenv import load_dotenv
    load_dotenv()

    # Set up Weights & Biases
    if track:
        print("Setting up Weights & Biases...")
        import wandb
        wandb.login(key=os.environ.get('WANDB_TOKEN'))

    print("Environment setup complete.")
