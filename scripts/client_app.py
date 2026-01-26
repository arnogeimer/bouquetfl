"""bouquetfl: A Flower / PyTorch app."""

import logging

from flwr.client import Client, ClientApp
from flwr.common import (
    Code,
    Context,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Status,
    parameters_to_ndarrays,
)

from bouquetfl.core.create_env import run_training_process_in_env
from bouquetfl.utils import power_clock_tools as pct
from bouquetfl.utils.filesystem import save_ndarrays

logger = logging.getLogger(__name__)

import torch

#####################################
########## Flower Client ############
#####################################


# Define Flower Client and client_fn
class FlowerClient(Client):
    def __init__(
        self,
        client_id: int,
    ) -> None:
        self.client_id = client_id
        self.num_examples = 1

    def fit(self, ins: FitIns) -> FitRes:
        # Save the global model parameters to a file to be loaded by trainer.py
        save_ndarrays(
            parameters_to_ndarrays(ins.parameters),
            f"checkpoints/global_params_round_{ins.config['server_round']}.npz",
        )

        status, parameters_updated = run_training_process_in_env(self.client_id, ins)

        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=self.num_examples,
            metrics={"client_id": self.client_id},
        )

    def evaluate(self, ins: EvaluateIns):
        from task import cifar100 as flower_baseline

        testset = flower_baseline.load_global_test_data()
        model = flower_baseline.get_model()
        ndarrays = parameters_to_ndarrays(ins.parameters)
        flower_baseline.ndarrays_to_model(model, ndarrays)
        loss, accuracy = flower_baseline.test(
            model=model,
            testloader=testset,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=loss,
            num_examples=self.num_examples,
            metrics={"accuracy": accuracy},
        )


def client_fn(context: Context) -> Client:
    # Return Client instance
    return FlowerClient(client_id=context.node_config["partition-id"]).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
