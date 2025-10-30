import torchvision
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (DirichletPartitioner, LinearPartitioner,
                                       SizePartitioner, SquarePartitioner)
from torch.utils.data import DataLoader

partitioner = SizePartitioner(
    partition_sizes=[10000, 10, 10, 10, 10, 10, 10, 10, 10, 10],
)
ds = FederatedDataset(
    dataset="uoft-cs/cifar100",
    partitioners={"train": partitioner},
    trust_remote_code=True,
)

transform_train = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomRotation(15),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761),
        ),
    ]
)


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [
        torchvision.transforms.ToTensor(transform_train(img)) for img in batch["img"]
    ]
    return batch


for i in range(10):
    dataset = ds.load_partition(partition_id=i)

    dataset = dataset.with_transform(apply_transforms)
    print(f"Partition {i}: {len(dataset)} samples")
    trainloader = DataLoader(
        dataset=dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=4
    )
