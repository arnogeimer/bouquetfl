import datasets
import numpy as np
import torch
import torchvision


class CUDA_VisionDataSet:
    def __init__(
        self,
        dataset: datasets.Dataset,
        transform: torchvision.transforms.Compose = torchvision.transforms.Compose([]),
        image_name: str = "image",
        target_name: str = "label",
    ):
        self.data = dataset[image_name]
        self.targets = dataset[target_name]

        self.image_name = image_name
        self.target_name = target_name

        self.train_data = torch.stack(
            [
                (
                    transform(entry).to_numpy()
                    if isinstance(entry, torch.Tensor)
                    else transform(
                        torchvision.transforms.functional.to_tensor(
                            entry.convert("RGB")
                        )
                    )
                    # We convert to RGB (on CPU) to handle grayscale images in RGB datasets
                )
                for entry in self.data
            ],
            dim=0,
        ).to("cuda")
        self.len = len(self.targets)
        self.targets = torch.tensor(
            (np.array(self.targets).reshape(self.len)), dtype=torch.long
        ).to("cuda")
        del dataset

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.train_data[idx], self.targets[idx]
