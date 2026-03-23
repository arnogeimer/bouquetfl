# BouquetFL

**BouquetFL** is a physical-layer emulation framework for **Federated Learning** built on top of [Flower](https://flower.ai). It simulates the full lifecycle of a worldwide federation — network transfers, hardware-constrained training, and failure modes like out-of-memory — on a single physical machine.

Everything physical is emulated: GPU clocks, CPU frequency, memory limits, upload/download times, and inter-country latency. You bring your own training task; BouquetFL handles the rest.

### A federation round in action

![BouquetFL round visualization](visuals/round_example.gif)

Each client downloads the global model, trains locally under its hardware constraints, and uploads the result — all with realistic timing derived from its assigned location and device profile.

---

## How It Works

BouquetFL is an overlay on Flower. For each simulated client, it:

1. **Assigns a hardware profile** (GPU, CPU, RAM) sampled from the [Steam Hardware Survey](https://store.steampowered.com/hwsurvey/)
2. **Assigns a network location** (country) with real-world upload/download speeds and inter-country ping
3. **Enforces hardware constraints** at the OS level: GPU memory/clock locking via `nvidia-smi`, CPU frequency via `cpupower`, RAM limits via cgroups
4. **Estimates network overhead** (model upload/download time) based on model size, link speed, and latency
5. **Handles failures** gracefully — OOM clients return the unmodified model

Clients run sequentially because hardware constraints (GPU clocks, CPU frequency) are global system settings.

---

## Quick Start

```bash
flwr run .
```

All federation-specific configuration lives in `pyproject.toml` — number of clients, rounds, model, learning rate, etc.

On first run, you will be prompted for your sudo password (needed for `cpupower` and `nvidia-smi`). It is stored in the system keyring for subsequent runs.

---

## Client Hardware Config

Profiles are either **auto-sampled** from the hardware database or **manually defined** in `federation_client_hardware.toml`:

```toml
[client_0]
gpu    = "GeForce GTX 1080"
cpu    = "Ryzen 3 3100"
ram_gb = 16
location = "Brazil"

[client_1]
gpu    = "GeForce RTX 3060"
cpu    = "Core i5-12400F"
ram_gb = 32
location = "Japan"
```

If no config file is present, BouquetFL samples realistic profiles automatically — constrained to hardware the host machine can actually emulate (you can't simulate a GPU more powerful than the one you have).

---

## Bring Your Own Task

BouquetFL is model- and task-agnostic. Add a task file to `task/` that provides:

- `get_model()` — return a PyTorch model
- `load_data(client_id, num_clients, num_workers, batch_size)` — return a DataLoader
- `train(model, trainloader, epochs, device, lr)` — training loop
- `test(model, testloader, device)` — evaluation loop

Set `experiment = "your_task"` in `pyproject.toml` and you're done. BouquetFL applies hardware constraints around your training code — no changes needed.

---

## Limitations

- Requires **NVIDIA GPU + CUDA** and **Ubuntu Linux**
- Clients execute **sequentially** (global hardware settings)
- Cannot emulate hardware **more powerful** than the host machine

---

## License

BouquetFL is licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).
