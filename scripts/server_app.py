"""bouquetfl: A Flower / PyTorch app."""

import os

from flwr.common import Context
from flwr.common.parameter import parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from torch.utils.tensorboard import SummaryWriter

from bouquetfl.utils.sampler import generate_hardware_config

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

experiment = "cifar10"
if experiment == "cifar10":
    from task import cifar10 as flower_baseline

# Create ServerApp
app = ServerApp()


@app.main()

def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Generate hardware profiles for clients
    if not os.path.exists("./config/federation_client_hardware.yaml"):
        generate_hardware_config(num_clients=context.run_config["num-clients"])

    arrays = ArrayRecord(flower_baseline.get_initial_state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")

def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    writer = SummaryWriter(f"logs/evaluate")
    # Load the model and initialize it with the received weights
    model = flower_baseline.get_model()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = flower_baseline.load_global_test_data()

    # Evaluate the global model on the test set
    test_loss, test_acc = flower_baseline.test(model, test_dataloader, device)
    writer.add_scalar("global accuracy", test_acc, server_round)
    writer.flush()
    # Return the evaluation metrics
    return MetricRecord({"loss": test_loss, "accuracy": test_acc})

