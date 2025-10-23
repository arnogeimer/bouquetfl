from bouquetfl.client_app import client_fn
import flwr as fl
from flwr.server.strategy import FedAvg

client_resources = {"num_cpus": 4, "num_gpus": 1}

from bouquetfl.data import cifar100 as flower_baseline

hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=2,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=FedAvg(
            initial_parameters=flower_baseline.get_initial_parameters(),
            evaluate_fn=flower_baseline.evaluate_fn,
        ),
        client_resources=client_resources,
    )
