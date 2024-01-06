# Imports
from BertFt.dataloader.model_data import get_model_data
from model.optimizer import getOptim
import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn
import torch.backends.mps
from torch.cuda import (
    max_memory_allocated,
    reset_peak_memory_stats,
    reset_max_memory_allocated,
    memory_allocated,
)

import pandas as pd
import weightwatcher as ww
import time
import os

import logging
import sys
from pathlib import Path
from collections import defaultdict

from distutils.util import strtobool

# other imports
# import datasets
# import evaluate
# from accelerate.logging import get_logger
from accelerate import Accelerator
import accelerate.utils

# from sklearn.model_selection import train_test_split

# from torch.utils.data import TensorDataset

from tqdm.auto import tqdm
from transformers import (
    set_seed,
    # get_scheduler,
)


parser = argparse.ArgumentParser(description="BERT Fine-Tuning")
# Parser Arguments and Defaults
parser.add_argument(
    "--savepath",
    type=str,
    default="/rscratch/tpang/kinshuk/RpMKin/misc/models",
    help="",
)
parser.add_argument("--epochs", type=int, default=20, help="")
parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="")
parser.add_argument("--task_name", type=str, default="cola", help="")
parser.add_argument(
    "--sortby", type=str, default="alpha", help="Use either of [alpha, layer, random]"
)
parser.add_argument("--max_length", type=int, default=128, help="")
parser.add_argument("--batch_size", type=int, default=32, help="")
parser.add_argument("--learning_rate", type=float, default=2e-5, help="")
parser.add_argument("--seed", type=int, default=7, help="")
parser.add_argument(
    "--freeze",
    type=lambda b: bool(strtobool(b)),
    nargs="?",
    const=False,
    default=True,
    help="",
)
parser.add_argument("--num_layers", type=int, default=0, help="")
parser.add_argument(
    "--alpha_ascending",
    type=lambda b: bool(strtobool(b)),
    nargs="?",
    const=False,
    default=False,
    help="",
)
parser.add_argument(
    "--slow_tokenizer",
    type=lambda b: bool(strtobool(b)),
    nargs="?",
    const=False,
    default=True,
    help="",
)
parser.add_argument(
    "--pad_to_max_length",
    type=lambda b: bool(strtobool(b)),
    nargs="?",
    const=False,
    default=True,
    help="",
)
parser.add_argument("--max_train_steps", type=int, default=1000, help="")
parser.add_argument("--grad_acc_steps", type=int, default=1, help="")
parser.add_argument(
    "--accelerate",
    type=lambda b: bool(strtobool(b)),
    nargs="?",
    const=False,
    default=False,
    help="",
)
parser.add_argument(
    "--add_layer_norm",
    type=lambda b: bool(strtobool(b)),
    nargs="?",
    const=False,
    default=False,
    help="",
)
parser.add_argument(
    "--verbose",
    type=lambda b: bool(strtobool(b)),
    nargs="?",
    const=False,
    default=True,
    help="",
)
parser.add_argument(
    "--debug",
    type=lambda b: bool(strtobool(b)),
    nargs="?",
    const=False,
    default=True,
    help="",
)
parser.add_argument(
    "--memlog",
    type=lambda b: bool(strtobool(b)),
    nargs="?",
    const=False,
    default=False,
    help="",
)
args = parser.parse_args()

# Control randomness
if args.verbose:
    print("SEED:", args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# accelerate.utils.set_seed(args.seed)
set_seed(args.seed)  # transformers

accelerator = Accelerator()

from transformers.utils.logging import (
    set_verbosity_error as transformers_set_verbosity_error,
)
from datasets.utils.logging import (
    set_verbosity_error as datasets_set_verbosity_error,
)

if not args.verbose:
    transformers_set_verbosity_error()
    datasets_set_verbosity_error()
    global _tqdm_active
    _tqdm_active = False

os.environ["TRANSFORMERS_CACHE"] = "/rscratch/tpang/kinshuk/cache"
cuda_device = torch.cuda.current_device()
reset_peak_memory_stats(device=cuda_device)
reset_max_memory_allocated(device=cuda_device)
start_memory = memory_allocated(device=cuda_device)


# Validation Loss (Classification)
def calc_val_loss(model, eval_dataloader, device):  # Done
    """_summary_

    Args:
        model (_type_): _description_
        eval_dataloader (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
    """
    loss = 0
    val_examples = 0
    correct = 0
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        input_len = len(batch["input_ids"])
        with torch.no_grad():
            outputs = model(
                # **batch,
                input_ids=batch["input_ids"].to(device),
                token_type_ids=batch["token_type_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            logits = outputs.logits
            _, predict = torch.max(logits, dim=1)

            correct += sum(predict == batch["labels"].to(device)).item()  # type: ignore

        loss += outputs.loss.item()
        val_examples += input_len
    if args.task_name == "stsb":
        return loss / len(eval_dataloader), 0
    return loss / len(eval_dataloader), correct / val_examples


# Training Loss
def calc_train_loss(  # Done
    args, model, optimizer, device, train_dataloader, eval_dataloader
):
    """_summary_

    Args:
        model (_type_): _description_
        optimizer (_type_): _description_
        device (_type_): _description_
        train_dataloader (_type_): _description_
        eval_dataloader (_type_): _description_

    Returns:
        _type_: _description_
    """
    model.train()
    num_all_pts = 0
    train_losses = []
    val_losses = []
    val_accs = []

    stats_path = os.path.join(args.savepath, "stats")
    Path(stats_path).mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    num_steps = args.epochs * len(train_dataloader)

    progress_bar = tqdm(
        range(num_steps),
        disable=not args.verbose,
    )

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        val_loss = 0
        tr_examples, tr_steps = 0, 0

        if 'lora' not in args.sortby.lower():
            # Save WeightWatcher Metrics
            watcher = ww.WeightWatcher(model=model)
            ww_details = watcher.analyze(min_evals=10)

        if not args.debug and 'lora' not in args.sortby.lower():
            ww_details.to_csv(os.path.join(stats_path, f"epoch_{epoch}.csv")) # type: ignore

        if epoch == 0 and 'lora' not in args.sortby.lower():
            # CHOOSING LAYERS TO TRAIN
            filtered = ww_details[ # type: ignore
                ww_details["longname"].str.contains("new_layer|embeddings") == False # type: ignore
            ]
            sortby = "alpha"
            if args.num_layers > len(filtered):
                args.num_layers = len(filtered)
            if "random" in (args.sortby).lower():
                train_names = random.sample(
                    filtered["longname"].to_list(), args.num_layers
                )
            else:
                if "alpha" in (args.sortby).lower():
                    sortby = "alpha"
                elif "layer" in (args.sortby).lower():
                    sortby = "layer_id"
                else:
                    sortby = "random"
                train_names = (
                    filtered.sort_values(by=[sortby], ascending=args.alpha_ascending)[
                        "longname"
                    ]
                    .iloc[: args.num_layers]
                    .to_list()
                )
            if args.verbose:
                print("Sorted by ", sortby)
                print("Training layers:", train_names)

            layer_to_train = []

            for layer in train_names:
                layer_to_train.append(layer + ".weight")
                layer_to_train.append(layer + ".bias")

                # Add Layer Norm
                if args.add_layer_norm:
                    if "output" in layer:
                        layer_to_train.append(
                            layer.replace("dense", "LayerNorm") + ".weight"
                        )
                        layer_to_train.append(
                            layer.replace("dense", "LayerNorm") + ".bias"
                        )

            layer_to_train = list(set(layer_to_train))

            # print("Final Training layers:", layer_to_train)

            for name, param in model.named_parameters():
                if name in layer_to_train:
                    if args.verbose:
                        print(f"Enabling {name} parameter")
                    param.requires_grad = True

        if args.verbose:
            print(f"===================================> Epoch {epoch+1}/{args.epochs}")
        # Training Loop
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(
                # **batch,
                input_ids=batch["input_ids"].to(device),
                token_type_ids=batch["token_type_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            train_loss += outputs.loss.item()
            # if args.accelerate:
            #     accelerator.backward(outputs.loss)
            # else:
            #     outputs.loss.backward()
            outputs.loss.backward()
            optimizer.step()
            tr_examples += len(batch["labels"])
            num_all_pts += len(batch["labels"])
            tr_steps += 1
            train_losses.append(train_loss / tr_steps)

            if not args.debug and 'lora' not in args.sortby.lower():
                # Saving Details of Frozen Layers
                freeze_dict = None
                if step in [0]:
                    freeze_dict = defaultdict(list)
                    for name, param in model.named_parameters():
                        freeze_dict["name"].append(name)
                        if param.grad is None:
                            freeze_dict["freeze_layer"].append(True)
                        elif torch.sum(param.grad.abs()).item() > 0:
                            freeze_dict["freeze_layer"].append(False)
                if freeze_dict is not None:
                    pd.DataFrame(freeze_dict).to_csv(
                        os.path.join(stats_path, f"freeze_{epoch}.csv")
                    )
            progress_bar.update(1)
            # if step >= 0.1 * len(train_dataloader) and args.task_name == 'wnli':
            #     break
        time_elapsed = (time.time() - start_time) / 60

        # Validation Loss
        val_loss, val_acc = calc_val_loss(model, eval_dataloader, device)
        if args.verbose:
            print(
                f"\nEpoch: {epoch+1}/{args.epochs}|Elapsed: {time_elapsed:.2f} mins|Val Loss: {val_loss:.4f}|Val Acc: {val_acc:.4f}"
            )
        val_losses.append(val_loss)
        val_accs.append(val_acc)

    return train_losses, val_losses, val_accs


def get_logger(path, fname):
    if not os.path.exists(path):
        os.mkdir(path)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_log_handler = logging.FileHandler(
        os.path.join(path, fname), mode="a"
    )  # 'a' for append
    stderr_log_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(file_log_handler)
    logger.addHandler(stderr_log_handler)
    formatter = logging.Formatter(
        "%(asctime)s;%(levelname)s;%(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_log_handler.setFormatter(formatter)
    stderr_log_handler.setFormatter(formatter)
    sys.stdout.flush()

    return logger


# Main
def main():
    if args.verbose:
        task_info = (
            f"\n\n\nTask to finetune: {args.task_name}\n\n\n"
            + f"alpha Decreasing: {not args.alpha_ascending}\n\n\n"
            + f"Layers to train: {args.num_layers}\n\n\n"
            + f"Train randomly: {'random' in args.sortby.lower()}\n\n\n"
        )
        print(task_info)

    log_info = (
        f"\n\n{args.task_name} "
        + f"{args.num_layers} Layers "
        + f"{args.sortby} "
        + f"ascending {args.alpha_ascending}"
    )
    if not args.verbose:
        transformers_set_verbosity_error()
        datasets_set_verbosity_error()
        global _tqdm_active
        _tqdm_active = False
    # Accelerator
    device = None

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # if args.accelerate:
    #     device = accelerator.device

    # Get Data
    model, train_dataloader, eval_dataloader = get_model_data(args)
    model.to(device)  # type: ignore

    if args.verbose:
        print(f"Training data size: {len(train_dataloader)}")
        print(f"Validation data size: {len(eval_dataloader)}")

    # Get Model and Optimizer
    # model = get_model(args=args, num_labels=num_labels, device=device)
    optimizer = getOptim(args, model, vary_lyre=False, factor=1)

    # Accelerator
    # if args.accelerate:
    #     model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    #         model, optimizer, train_dataloader, eval_dataloader
    #     )

    # Get Initial Validation Loss
    i_val_loss, i_val_acc = calc_val_loss(model, eval_dataloader, device)
    if args.verbose:
        print(
            f"\nEpoch 0/{args.epochs} | Val Loss: {i_val_loss:.2f} | Val Acc: {i_val_acc:.2f}"
        )

    # Get Training Loss
    train_loss, val_loss, val_acc = calc_train_loss(
        args=args,
        model=model,
        optimizer=optimizer,
        device=device,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )

    val_loss = [i_val_loss] + val_loss
    val_acc = [i_val_acc] + val_acc
    base = {
        "train_loss_base": train_loss,
        "val_loss_base": val_loss,
        "val_acc_base": val_acc,
    }

    if args.memlog:
        end_memory = memory_allocated(device=cuda_device)
        peek_memory = max_memory_allocated(device=cuda_device)
        mempath = f"/rscratch/tpang/kinshuk/RpMKin/bert_ft/GLUE/trainseed_{args.seed}" + \
                f"/task_{args.task_name}/{args.sortby}" #_asc_{args.alpha_ascending}"
        Path(mempath).mkdir(parents=True, exist_ok=True)
        logger = get_logger(mempath, "memlog.log")
        logger.info(log_info)
        logger.info(
            f"\nMemory usage before: {start_memory} bytes\nMemory usage after: {int((end_memory/1024)/1024)}MB"
        )
        logger.info(f"\nPeak Memory usage: {int((peek_memory/1024)/1024)}MB\n\n")

    if args.debug and args.verbose:
        print("\n--> Debug Mode <--")
        print("\nTrain Loss:")
        print(*[train_loss[i] for i in range(0, len(train_loss), args.batch_size)])
        print("\nVal Loss:\n", val_loss)
        print("\nVal Acc:\n", val_acc)
    else:
        # Save the data
        Path(args.savepath).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(args.savepath, "baseline.npy"), base)  # type: ignore


if __name__ == "__main__":
    main()

'''
# overrode_max_train_steps = False
#     num_steps_per_epoch = math.ceil(
#         len(train_dataloader) / args.gradient_acc_steps
#     )
#     if args.max_train_steps is None:
#         args.max_train_steps = args.epochs * num_steps_per_epoch
#         overrode_max_train_steps = True
#     if args.task_name is None:
#         metric = evaluate.load(
#             "glue",
#             args.task_name,
#             experiment_id=str(args.seed) + str(args.num_layers) + str(args.alpha_ascending),
#         )
#     else:
#         metric = evaluate.load("accuracy")


# Custom Preprocessing | WORK IN PROGRESS
def preprocessing(input_text, tokenizer):
    """
    Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
    - input_ids: list of token ids
    - token_type_ids: list of token type ids
    - attention_mask: list of indices (0,1) specifying which tokens should considered
    by the model (return_attention_mask = True).
    """
    return tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=32,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors="pt",
    )


# Get Custom Dataloaders | WORK IN PROGRESS
def get_dataloaders(args, df, val_ratio=0.2, fract=0.1):
    labels = df["label"].values
    train_ix, val_ix = train_test_split(
        np.arange(len(labels)),
        test_size=val_ratio,
        stratify=labels,
        random_state=args.seed,
    )

    text = df.text.values
    labels = df.label.values
    truncate_text = False

    if truncate_text:
        text = text[: int(len(text) * fract)]
        labels = labels[: int(len(labels) * fract)]

    tokenizer = BertTokenizer.from_pretrained(  # Done
        args.model_name,
        do_lower_case=True,
        use_fast=not args.slow_tokenizer,
    )
    token_ids = []
    attention_masks = []
    for sample in text:
        en_dict = preprocessing(sample, tokenizer)
        token_ids.append(en_dict["input_ids"])
        attention_masks.append(en_dict["attention_mask"])

    token_ids = torch.cat(token_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    train_ix = train_ix[: int(len(train_ix) * fract)]

    train_dataset = TensorDataset(
        token_ids[train_ix], attention_masks[train_ix], labels[train_ix]
    )

    val_dataset = TensorDataset(
        token_ids[val_ix], attention_masks[val_ix], labels[val_ix]
    )

    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,  # type: ignore
        shuffle=True,
        batch_size=args.batch_size,
    )

    val_dataloader = DataLoader(
        val_dataset,  # type: ignore
        batch_size=args.batch_size,
    )

    return train_dataloader, val_dataloader
'''
