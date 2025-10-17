# 💐 bouquetfl: Federated Hardware Simulation with Flower 💐

Simulate heterogeneous client hardware in **federated learning**, all on a single machine.  
This framework allows you to mimic clients with limited CPU, GPU, and RAM resources, enabling research into how federated learning behaves on lower-grade hardware without needing actual diverse devices.  

**Bouquet** - from the French word for “a bunch of flowers” - reflects how this project gathers many different *“flowers”* (clients built on the [Flower](https://flower.dev/) framework) into a single federated learning environment, each representing a unique hardware profile.



## 📝 Introduction
Federated learning often assumes diverse client devices, but most research setups use uniform, high-end hardware.  
This project provides a way to **simulate real-world device heterogeneity**:  

- We maintain a **large list of consumer-grade GPUs and CPUs**.  
- Clients are spawned as **subprocesses**, each constrained by the specs of a chosen GPU–CPU combination.  
- Each subprocess applies resource limits (GPU threads, CPU cores, RAM) to mimic that hardware profile.  

This enables federated learning experiments that reflect a more realistic hardware distribution without requiring dozens of different physical devices.  



## ✨ Features
- Simulate **CPU, GPU, and RAM constraints** for Flower clients.  
- Run **different hardware profiles** within the same federation.  
- Supports **multiple experiments** across different domains.


## 📦 Dependencies
- Python **3.10+**  
- [PyTorch 2.7.0](https://pytorch.org/)  
- [Flower 1.20.0](https://flower.dev/)  
- [flwr-datasets 0.5.0](https://flower.dev/docs/datasets.html)
- [nvidia-smi](https://developer.nvidia.com/system-management-interface)
- [Nvidia Multi-Processor Service](https://docs.nvidia.com/deploy/mps/index.html) capable GPU (Volta+, i.e. GeForce GTX 16 series and after)
- Root access (`sudo`) for simulating hardware restrictions  



## 🖥️ Example Hardware Profiles
This project can simulate a wide range of common consumer hardware from:  

- **NVIDIA GeForce GTX & RTX series**
- **AMD Radeon RX 500 series**
- **Intel Arc Series**

### CPUs
- **Intel Core**
- **AMD Ryzen, EPYC & Athlon**

📂 See the full [GPU list](bouquetfl/hardwareconf/gpus.csv) and [CPU list](bouquetfl/hardwareconf/cpus.csv) for all supported profiles.  

Each simulated client process enforces limits (cores, threads, memory, GPU scheduling) to approximate the chosen hardware.  


## 📊 Experiments
We provide experiment setups for:  

- **Image classification**  
  - Datasets: [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), [Tiny-ImageNet](https://www.kaggle.com/c/tiny-imagenet)  
  - Models: Any suitable architecture from [timm](https://github.com/huggingface/pytorch-image-models)  

- **Large language model finetuning**  
  - Dataset: [Alpaca-GPT4](https://huggingface.co/datasets/vicgalle/alpaca-gpt4)  
  - Model: [OpenLLaMA 3B v2](https://huggingface.co/openlm-research/open_llama_3b_v2)  


## ⚠️ Limitations
- Cannot simulate **better hardware** than the host device (e.g. a GTX 2060 cannot simulate an RTX 3080).  
- Requires **sudo** access.  
- Currently supports **only one client at a time**.  
- This is a **simulation**, not a real hardware replacement.
- Only supports **Nvidia hardware** (for now), due to reliance on nvidia-smi.

## ⚙️ How It Works

This project simulates a wide variety of client hardware profiles **by programmatically constraining local system resources** during the training of each [Flower](https://flower.dev/) federated client.  
All constraints are applied at runtime so that multiple, heterogeneous “virtual clients” can be successively emulated on a single machine.

### CPU
* **Frequency capping** – Uses [`cpupower`](https://linux.die.net/man/1/cpupower) to temporarily set the CPU’s maximum clock (`-u`) and minimum clock (`-d`) during training.  
* **Core limitation** – Reduces the number of CPU cores the client can effectively use by controlling the number of **DataLoader workers** in the training loop.

### RAM
* **Memory ceiling** – Each training process is launched through:
  ```bash
  systemd-run --user --scope -p MemoryMax=<limit> …
  ```
  so the Linux systemd cgroup enforces a hard upper bound on RAM usage.

### GPU

* **Memory limit** – Calls:

  ```python
  torch.cuda.set_per_process_memory_fraction(fraction, device)
  ```
  to cap the fraction of GPU memory available to the client.

* **Clock speed lock** – Uses:
  ```bash
  nvidia-smi -lock-gpu-clocks <min>,<max>        # lock GPU graphics clocks
  nvidia-smi -lock-memory-clocks <min>,<max>        # lock GPU memory clocks
  ```
  to fix GPU and memory clocks at the desired frequencies.

* **Core usage fraction** – Launches the training in a subprocess with:
  ```python
  os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = "<percentage>"
  ```
  so that **NVIDIA Multi-Process Service (MPS)** restricts the fraction of CUDA cores allocated relative to the total cores of the host GPU.

Note: All these constraints are runtime-only—they simulate weaker hardware but cannot exceed the capabilities of the physical machine.

## 🗺️ Roadmap
- Add **timekeeping** to record and showcase how long each client configuration takes to complete training and communication.
- Add **location simulation** (to model upload/download latency).  
- Support **parallel client spawning** (multiple clients at once).
- Add **mobile devices** (e.g., phones, Raspberry Pi, etc.) as available options.
- Allow users to **add custom hardware specifications** for new CPU/GPU profiles.
- Support for **AMD GPUs**.
- Add **per-client resource monitoring** for in-depth analysis.



## 📜 License
This project is licensed under the **MIT License** (open and permissive).  



## 🙏 Acknowledgements
- [Flower](https://flower.dev/) for federated learning framework  
- [PyTorch](https://pytorch.org/)  
- [timm](https://github.com/huggingface/pytorch-image-models)  
























