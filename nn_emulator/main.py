# main.py

from config import redshift_configs
from data_processing import get_train_test_data
from train import train_model, check_test_distribution
from model import NNInterpolator
import torch
import argparse

def main(target_redshift):
    """
    Run the full pipeline for each redshift configuration.
    """

    # Get the configuration for the specified redshift
    config = redshift_configs[target_redshift]
    print(f"Processing redshift {target_redshift}...")
    
    print(f"Preprocessing data for redshift {target_redshift}...")
    get_train_test_data(
            z=target_redshift,
            base_path=config["base_path"],
            batch_size=config["batch_size"],
            patience=config["patience"],
            patience_scheduler=config["patience_scheduler"]
        )

    # Step 2: Train the model
    print(f"Training model for redshift {target_redshift}...")
    model = NNInterpolator(
        input_size=6,
        output_size=16,
        hidden=config["hidden"],
        learning_rate=config["learning_rate"],
        act=torch.nn.ReLU,
        patience=config["patience"],
        patience_scheduler=config["patience_scheduler"],
        base_path=config["base_path"]
    )
    train_model(config, target_redshift)

    # Step 3: Inspect the test distribution
    print(f"Inspecting test distribution for redshift {target_redshift}...")
    check_test_distribution(model, target_redshift, config["base_path"], fold="train")

    print(f"Completed processing for redshift {target_redshift}.")



if __name__ == "__main__":
    # Define argument parser
    parser = argparse.ArgumentParser(description="Run the pipeline for a specified redshift.")
    parser.add_argument(
        "--redshift",
        type=float,
        required=True,
        help="The redshift value to process"
    )

    # Parse arguments
    args = parser.parse_args()

    # Call main with the provided redshift
    main(args.redshift)
