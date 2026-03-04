from bouquetfl.core.emulation_engine import run_emulation

config = {
    # worker identity
    "task":              "task/cifar10.py",
    "client_id":         0,               # uses client_0 from federation_client_hardware.yaml
    # federation settings
    "num-clients":       5,
    "num-server-rounds": 3,
    "server-round":      1,
    # training hyperparameters
    "batch-size":        256,             # keep small for a quick smoke test
    "local-epochs":      10,
    "learning-rate":     0.01,
}

timing, state_dict = run_emulation(
    config=config,
    hardware_profile = {"gpu": "GeForce RTX 4060", "cpu": "Ryzen 7 5825U", "ram_gb": 16},
    input_params_path=None,  # uses cifar10.get_initial_state_dict()
    output_params=True,
)

print("returned state_dict:", type(state_dict))
if state_dict:
    first_key = next(iter(state_dict))
    print(f"  first layer '{first_key}': shape {state_dict[first_key].shape}")
print("timing:", timing)