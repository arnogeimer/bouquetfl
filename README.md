# 💐 BouquetFL

**BouquetFL** is a framework for simulating *heterogeneous client hardware* in **Federated Learning** using the [Flower](https://flower.ai) framework.  
It allows researchers to emulate clients with different CPU, GPU, and RAM capabilities **on a single physical machine** by enforcing hardware-level resource constraints at runtime.

BouquetFL is designed for studying realistic cross-device federated learning scenarios—where client devices differ widely in computational power—without requiring access to large, heterogeneous hardware testbeds. Each simulated client runs sequentially under a configurable hardware profile, enabling controlled and reproducible experimentation.

---

## System Dependencies

BouquetFL relies on several **system-level tools** to enforce hardware constraints. These must be installed outside of Python.

### Operating System
- **Ubuntu Linux** (tested on 22.04 / 24.04)

### Required System Tools
- **sudo access** (required to control hardware settings)
- **NVIDIA GPU + CUDA**
  - `nvidia-smi` (comes with NVIDIA drivers)
- **cpupower** (CPU frequency control)
- **systemd** (for memory cgroup limits)
- **uv** (Python project runner)

> Python dependencies (e.g., `flwr`, `torch`) are managed automatically via the project configuration.

---

## Installation

### 1. Install system packages

```bash
sudo apt update
sudo apt install -y linux-tools-common linux-tools-generic cpupower systemd
```

### 2. Install NVIDIA drivers

Follow NVIDIA’s official instructions for your GPU and Ubuntu version.

Verify the installation:
```bash
nvidia-smi
```

### 3. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your shell and verify:
```bash
uv --version
```

### Running BouquetFL

From the project root, run:
```bash
flwr run .
```

BouquetFL integrates directly into the standard Flower workflow.

### First Run: sudo Password Handling

BouquetFL applies hardware limits using system-level tools such as cpupower and nvidia-smi, which require elevated privileges.

*On the first run, you will be prompted for your sudo password.

*The password is stored securely using the system keyring.

*Subsequent runs will not prompt again.

### Example Hardware Configuration

Client hardware profiles are defined using a YAML file.

```yaml
client_0:
  cpu: Ryzen 3 3100
  gpu: GeForce GTX 1080
  ram_gb: 16
```

BouquetFL uses these profiles to enforce corresponding CPU, GPU, and memory limits when spawning each client.

### Notes & Limitations

Clients are executed sequentially due to global hardware settings.

BouquetFL cannot emulate hardware more powerful than the host machine.

GPU support currently requires NVIDIA hardware.
