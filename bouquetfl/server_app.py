"""bouquetfl: A Flower / PyTorch app."""

from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from bouquetfl import power_clock_tools as pct

experiment = "cifar100"
if experiment == "cifar100":
    from bouquetfl.data import cifar100 as flower_baseline


pct.reset_all_limits()


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    def on_fit_client_config_fn(server_round: int) -> dict:
        """Return training configuration dict for each round."""
        client_config = {
            "server_round": server_round,
            "num_rounds": num_rounds,
            # You can add more configuration parameters here
        }
        return client_config

    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    initial_parameters = flower_baseline.get_initial_parameters()

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=initial_parameters,
        on_fit_config_fn=on_fit_client_config_fn,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
