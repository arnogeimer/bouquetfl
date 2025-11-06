import random
from typing import List

from datasets import Dataset, concatenate_datasets, load_from_disk
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import GroupedNaturalIdPartitioner
from transformers import WhisperProcessor

fds = None  # Cache FederatedDataset
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")


def load_data(
    partition_id: int,
    remove_cols: List[str],
):
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = GroupedNaturalIdPartitioner(
            partition_by="speaker_id", group_size=5
        )
        fds = FederatedDataset(
            dataset="speech_commands",
            subset="v0.02",
            partitioners={"train": partitioner},
            trust_remote_code=True,
        )

    partition = fds.load_partition(partition_id)

    encoding_fn = get_encoding_fn(processor)

    remove_cols = remove_cols.split(",")
    partition = partition.map(encoding_fn, num_proc=2, remove_columns=remove_cols)

    # Now let's add some _silence_ training examples (add 10% of total examples in this client's data)
    partitioner = fds.partitioners["train"]
    ratio_silences_for_client = 0.1 * (len(partition) / len(partitioner.dataset))
    silence_dataset = prepare_silences_dataset(
        partitioner.dataset, ratio_silences_for_client
    )
    if len(silence_dataset) > 0:
        silence_enc = silence_dataset.map(encoding_fn)
        partition = concatenate_datasets([partition, silence_enc])

    return partition


def load_data_from_disk(data_path):
    """Load ddata from a partition explicitly saved to disk."""
    return load_from_disk(data_path)


def get_encoding_fn(processor):
    """Return a function to use to pre-process/encode the SpeechCommands dataset.

    We are working with the 12classes version of this dataset, therefore we need to do
    some reassignment of labels.
    """

    def prepare_dataset(batch):
        audio = batch["audio"]
        data = {}
        data["data"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
        ).input_features

        # All unknown keywords are assigned label 11. The silence clips get assigned label 10
        # In this way we have 12 classes with labels 0-11
        data["targets"] = (
            11
            if batch["is_unknown"]
            else (10 if batch["label"] == 35 else batch["label"])
        )
        return data

    return prepare_dataset


def prepare_silences_dataset(train_dataset, ratio_silence: float = 0.1) -> Dataset:
    """Generate silences for the train set.

    One of the classes in the SpeechCommands datatset is `silence`. However, the dataset
    does not include clips of silence. It does however include 5 long files with
    different background sounds. The taks of this function is to extract several
    (defined by `ratio_silence`) one-second long clips from those background audio
    files. Later, those audio clips will be included into the training set.
    """
    # Retrieve original silence audio clips
    silences = train_dataset.filter(lambda x: x["label"] == 35)
    # Figure out how many to add
    num_silence_total = int(len(train_dataset) * ratio_silence)
    # Num new entries per background noise clip
    num_silence_per_bkg = num_silence_total // len(silences)

    silence_to_add = []
    for sil in silences:
        sil_array = sil["audio"]["array"]
        sr = sil["audio"]["sampling_rate"]
        # print(f"Extracting audio from: {sil['file']} ...")
        for _ in range(num_silence_per_bkg):
            random_offset = random.randint(0, len(sil_array) - sr - 1)
            sil_array_crop = sil_array[random_offset : random_offset + sr]

            entry = sil
            silence_to_add.append(entry)
            silence_to_add[-1]["audio"]["array"] = sil_array_crop

    return Dataset.from_list(silence_to_add)


import time

time.sleep(5)
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from torch.utils.data import DataLoader
from whisper_example.dataset import load_data, load_data_from_disk
from whisper_example.model import (
    construct_balanced_sampler,
    get_model,
    get_params,
    set_params,
    train_one_epoch,
)

torch.set_float32_matmul_precision(
    "high"
)  #  If “high” or “medium” are set then the TensorFloat32 is used

og_threads = torch.get_num_threads()


class WhisperFlowerClient(NumPyClient):
    """A Flower client that does trains a classification head attached to the encoder of
    a Whisper-tiny encoder for Keyword spotting."""

    def __init__(
        self,
        trainset,
        batch_size: int,
        num_classes: int,
        disable_tqdm: bool,
        compile: bool,
    ):
        self.disable_tqdm = disable_tqdm
        self.batch_size = batch_size
        self.trainset = trainset.with_format("torch", columns=["data", "targets"])

        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        self.encoder, self.classifier = get_model(self.device, num_classes, compile)

    def fit(self, parameters, config):
        """Do on-device training.

        Here the client receives the parameters of the classification head from the
        server. Then trains that classifier using the data that belongs to this client.
        Finally, The updated classifier is sent back to the server for aggregation.
        """

        # Apply the classifier parameters to the model in this client
        set_params(self.classifier, parameters)

        # construct sampler in order to have balanced batches
        sampler = None
        if len(self.trainset) > self.batch_size:
            sampler = construct_balanced_sampler(self.trainset)

        # Construct dataloader
        train_loader = DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            sampler=sampler,
            drop_last=True,
        )

        # Define optimizer and criterion
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.001)

        # Don't train if partition is very small
        run_training = len(train_loader) > 1
        metrics = {"trained": run_training}  # will be used for metrics aggregation
        if run_training:
            # Train
            avg_loss, avg_acc = train_one_epoch(
                self.encoder,
                self.classifier,
                optimizer,
                criterion,
                train_loader,
                self.device,
                disable_tqdm=self.disable_tqdm,
            )
            metrics = {**metrics, "loss": avg_loss, "accuracy": avg_acc}

        # Return local classification head and statistics
        return get_params(self.classifier), len(train_loader.dataset), metrics


def client_fn(context: Context):

    partition_id = context.node_config["partition-id"]
    num_classes = context.run_config["num-classes"]
    batch_size = context.run_config["batch-size"]
    disable_tqdm = context.run_config["disable-tqdm"]
    compile_model = context.run_config["compile-model"]

    # Some systems seem to need this, else .map stages will hang
    # Doesn't seem to be required on macOS; but it's on Ubuntu
    # even if the latter has more CPUs...
    # ! Open a PR if you know how to improve this!
    og_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    partition = load_data(
        partition_id=partition_id,
        remove_cols=context.run_config["remove-cols"],
    )

    torch.set_num_threads(og_threads)

    return WhisperFlowerClient(
        partition, batch_size, num_classes, disable_tqdm, compile_model
    ).to_client()


app = ClientApp(client_fn=client_fn)


"""whisper_example: A Flower / PyTorch app with OpenAi's Whisper."""

from collections import OrderedDict
from typing import List

import numpy as np
import torch
from flwr.common import NDArrays
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration


def get_model(device, num_classes, compile: bool = True):
    """Create model: Whisper-tiny Encoder + classification head."""
    encoder = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-tiny"
    ).get_encoder()
    encoder = encoder.to(device)
    if compile:
        encoder = torch.compile(encoder)

    # This classification head is 782K parameters
    # This is the only part of the model that is trained in federation
    classifier = torch.nn.Sequential(
        torch.nn.Conv1d(1500, 128, kernel_size=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(1),
        torch.nn.Linear(128 * 384, num_classes),
    ).to(device)
    return encoder, classifier


def set_params(model: torch.nn.ModuleList, params: List[NDArrays]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_params(module: torch.nn.ModuleList):
    return [val.cpu().numpy() for _, val in module.state_dict().items()]


class RunningAvg:
    def __init__(self):
        self.n = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.n += 1

    def __call__(self):
        return self.total / self.n


def construct_balanced_sampler(trainset):
    hist, _ = np.histogram(trainset["targets"], bins=12)
    # Mask of non-zeros
    hist_mask = hist > 0
    w_per_class = len(trainset) / (
        hist + 1
    )  # avoid dividing by zeros  # doesn't have to add up to 1 (relative is what matters)
    w_per_class += 1  # needed in case trainset has very few samples
    # Apply mask so we don't attempt sampling classes that aren't present
    w_per_class *= hist_mask
    w_ss = [w_per_class[t] for t in trainset["targets"]]
    return WeightedRandomSampler(w_ss, len(w_ss))


def train_one_epoch(
    model,
    classifier,
    optimizer,
    criterion,
    dataloader,
    device,
    disable_tqdm: bool = False,
):
    """Train the classification head.

    This is a very standard looking way of training PyTorch models.
    """
    model.eval()
    classifier.train()
    classifier.to(device)
    loss_avg, acc_avg = RunningAvg(), RunningAvg()
    avg_loss, avg_acc = 0.0, 0.0
    with tqdm(total=len(dataloader.dataset), disable=disable_tqdm) as t:
        for b in dataloader:
            optimizer.zero_grad()
            data = b["data"].squeeze().to(device)
            # print(data.shape)
            labels = b["targets"].to(device)
            with torch.no_grad():
                res = model(data)[0]

            resres = classifier(res)

            loss = criterion(resres.float(), labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(resres.data, 1)
            correct = (predicted == labels).sum().item()
            acc = correct / data.shape[0]
            loss_ = loss.cpu().item()

            loss_avg.update(loss_)
            acc_avg.update(acc)

            t.update(data.shape[0])
            avg_loss, avg_acc = loss_avg(), acc_avg()
            t.set_postfix({"avg_loss": f"{avg_loss:.4f}", "avg_acc": f"{avg_acc:.4f}"})

    return avg_loss, avg_acc


def eval_model(model, classifier, criterion, dataloader, device):
    """Evaluate a model on a validation/test set.

    This is a very normal looking way of doing this with PyTorch.
    """
    model.eval()
    classifier.eval()
    classifier.to(device)
    correct = 0
    loss_ = 0
    total = 0
    with torch.no_grad():
        for b in dataloader:
            data = b["data"].squeeze().to(device)
            # print(data.shape)
            labels = b["targets"].to(device)
            res = model(data)[0]
            resres = classifier(res)

            loss = criterion(resres.float(), labels)
            _, predicted = torch.max(resres.data, 1)
            correct += (predicted == labels).sum().item()
            total += data.shape[0]
            loss_ += loss.cpu().item()

    accuracy = correct / total
    loss = loss_ / total

    return loss, accuracy
