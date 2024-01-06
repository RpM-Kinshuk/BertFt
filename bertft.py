# Imports
import os
import torch
import random
import argparse
import numpy as np
# import accelerate.utils
from pathlib import Path
import torch.backends.mps
import torch.backends.cudnn
from torch.cuda import (
    max_memory_allocated,
    reset_peak_memory_stats,
    reset_max_memory_allocated,
    memory_allocated,
)
from transformers import set_seed
# from accelerate import Accelerator
from distutils.util import strtobool
from model.optimizer import getOptim
from traineval.eval import calc_val_loss
from traineval.train import calc_train_loss
from dataloader.logger import get_logger
from dataloader.model_data import get_model_data
from transformers.utils.logging import (
    set_verbosity_error as transformers_vb_err,
)
from datasets.utils.logging import (
    set_verbosity_error as datasets_vb_err,
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

# Set Cache Directory
cache_dir = "/rscratch/tpang/kinshuk/cache"

# Memory Log Path
mempath = (
    f"/rscratch/tpang/kinshuk/RpMKin/bert_ft/GLUE/trainseed_{args.seed}"
    + f"/task_{args.task_name}/{args.sortby}"  # _asc_{args.alpha_ascending}"
)


def main():
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    cuda_device = torch.cuda.current_device()

    # Control randomness
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # accelerate.utils.set_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seed(args.seed)  # transformers seed

    # Memory Stats Initialization
    reset_peak_memory_stats(device=cuda_device)
    reset_max_memory_allocated(device=cuda_device)
    start_memory = memory_allocated(device=cuda_device)

    if args.verbose:
        print("SEED:", args.seed)
        task_info = (
            f"\n\n\nTask to finetune: {args.task_name}\n\n\n"
            + f"alpha Decreasing: {not args.alpha_ascending}\n\n\n"
            + f"Layers to train: {args.num_layers}\n\n\n"
            + f"Train randomly: {'random' in args.sortby.lower()}\n\n\n"
        )
        print(task_info)
    else:
        datasets_vb_err()
        transformers_vb_err()
        global _tqdm_active
        _tqdm_active = False

    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # if args.accelerate:
    #     device = accelerator.device

    # Get Model, Data, Optimizer
    model, train_dataloader, eval_dataloader = get_model_data(args, cache_dir)
    model.to(device)  # type: ignore
    optimizer = getOptim(args, model, vary_lyre=False, factor=1)

    if args.verbose:
        print(f"Training data size: {len(train_dataloader)}")
        print(f"Validation data size: {len(eval_dataloader)}")

    # Accelerator
    # if args.accelerate:
    #     model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    #         model, optimizer, train_dataloader, eval_dataloader
    #     )

    # Get Initial Validation Loss
    i_val_loss, i_val_acc = calc_val_loss(args, model, eval_dataloader, device)
    if args.verbose:
        print(
            f"\nEpoch 0/{args.epochs} "
            + f"| Val Loss: {i_val_loss:.2f} "
            + f"| Val Acc: {i_val_acc:.2f}"
        )

    # Train and get Losses
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
        log_info = (
            f"\n\n{args.task_name} "
            + f"{args.num_layers} Layers "
            + f"{args.sortby} "
            + f"ascending {args.alpha_ascending}"
        )
        end_memory = memory_allocated(device=cuda_device)
        peek_memory = max_memory_allocated(device=cuda_device)
        Path(mempath).mkdir(parents=True, exist_ok=True)
        logger = get_logger(mempath, "memlog.log")
        logger.info(log_info)
        logger.info(
            f"\nMemory usage before: {(start_memory/1024)/1024}MB\n"
            + f"Memory usage after: {(end_memory/1024)/1024}MB"
        )
        logger.info(f"\nPeak Memory usage: {(peek_memory/1024)/1024}MB\n\n")

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
