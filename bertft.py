r"""Done
num_layers = "0 1 2 3 4 5 6 8 10 12 18 24 30 36 72 74"
task_list = "cola sst-2 mrpc sts-b qqp mnli qnli rte wnli"

for task in $task_list
do
    for layers in $num_layers
    do
        python bertft.py \
            --savepath /models \
            --epochs 20 \
            --model_name roberta-large \
            --task_name $task \
            --max_length 512 \
            --batch_size 32 \
            --learning_rate 2e-5 \
            --seed 5 \
            --freeze_bert True \
            --num_layers $layers \
            --alpha_ascending False \
            --slow_tokenizer True \
            --pad_to_max_length False \
            --max_train_steps 1000 \
    done
done
"""

import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn
from sklearn.model_selection import train_test_split

import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput

import pandas as pd
import weightwatcher as ww
import time
import os
import math
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Optional, Tuple, Union

# others
import datasets
import evaluate
from accelerate import Accelerator
from accelerate.logging import get_logger
import accelerate.utils
from datasets import load_dataset

from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import copy
from transformers import (
    BertTokenizer,
    BertPreTrainedModel,
    BertModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    set_seed,
    get_scheduler,
)


task_to_keys = {  # Done
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

parser = argparse.ArgumentParser(description="BERT Fine-Tuning")

parser.add_argument("--savepath", type=str, default="/models", help="")
parser.add_argument("--epochs", type=int, default=20, help="")
parser.add_argument("--model_name", type=str, default="roberta-large", help="")
parser.add_argument("--task_name", type=str, default="cola", help="")
parser.add_argument("--max_length", type=int, default=512, help="")
parser.add_argument("--batch_size", type=int, default=32, help="")
parser.add_argument("--learning_rate", type=float, default=2e-5, help="")
parser.add_argument("--seed", type=int, default=5, help="")
parser.add_argument("--freeze_bert", type=bool, default=True, help="")
parser.add_argument("--num_layers", type=int, default=74, help="")
parser.add_argument("--alpha_ascending", type=bool, default=False, help="")
parser.add_argument("--slow_tokenizer", type=bool, default=True, help="")
parser.add_argument("--pad_to_max_length", type=bool, default=False, help="")
parser.add_argument("--max_train_steps", type=int, default=None, help="")

args = parser.parse_args()

print("SEED:", args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
accelerate.utils.set_seed(args.seed)
set_seed(args.seed)  # transformers

accelerator = Accelerator()


class BERTFT(BertPreTrainedModel):  # Done
    """
    BERT Fine-Tuning on GLUE tasks
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.new_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.relu = nn.ReLU()
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        Feed-forward function for the bert ft model
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        intermediate_output = self.new_layer(pooled_output)
        intermediate_output = self.relu(intermediate_output)
        intermediate_output = self.dropout(intermediate_output)
        logits = self.classifier(intermediate_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    #  We are doing regression
                    self.config.problem_type = "regression"
                elif (
                    self.num_labels > 1
                    and labels.dtype == torch.long
                    or labels.dtype == torch.int
                ):
                    # We are doing multi-class classification
                    self.config.problem_type = "single_label_classification"
                else:
                    # We are doing multi-label classification
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


config = BERTFT.from_pretrained(  # WORK IN PROGRESS
    args.model_name,
    num_labels=num_labels,
    finetuning_task=args.task_name,
    cache_dir=args.savepath,
)


def getCustomParams(model):  # Done
    new_params = []
    pre_trained = []
    for name, val in model.named_parameters():
        if "new_layer" in name or "classifier" in name:
            new_params.append(val)
        else:
            pre_trained.append(val)
    return new_params, pre_trained


def getOptim(model, vary_lyre, factor=1):  # Done
    if vary_lyre:
        new_params, pre_params = getCustomParams(model)
        return torch.optim.AdamW(
            [
                {"params": new_params, "lr": args.learning_rate * factor},
                {"params": pre_params, "lr": args.learning_rate},
            ],
        )
    else:
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
        )


def get_model(args, device):  # Done
    model = BERTFT.from_pretrained(
        args.model_name,
        # num_labels=num_labels,
        # finetuning_task=args.task_name,
        # cache_dir=args.savepath,
        config=config,
    )
    model.to(device)  # type: ignore
    if args.freeze_bert:
        print("Freezing BERT")
        for name, param in model.named_parameters():  # type: ignore
            if "classifier" in name or "new_layer" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    else:
        print("Defreezing BERT")
        for param in model.parameters():  # type: ignore
            param.requires_grad = True
    return model


def calc_val_loss(model, eval_dataloader, device):  # Done
    loss = 0
    val_examples = 0
    correct = 0
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        batch = tuple(t.to(device) for t in batch)
        """
        Assuming batch = (input_ids, attention_mask, labels)
        """
        input_len = len(batch[2])
        with torch.no_grad():
            outputs = model(
                **batch,
                token_type_ids=None,
                attention_mask=batch[1].to(device),
                labels=batch[2].to(device),
            )
            logits = outputs.logits
            _, predict = torch.max(logits, dim=1)

            correct += sum(predict == batch[2]).item()  # type: ignore

        loss += outputs.loss.item() * input_len
        val_examples += input_len
    return loss / val_examples, correct / val_examples


def calc_train_loss(args, model,  # Done
                    optimizer, device, 
                    train_dataloader, eval_dataloader
):
    num_all_pts = 0

    train_losses = []
    val_losses = []
    val_accs = []

    stats_path = os.path.join(args.savepath, "stats")
    Path(stats_path).mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator()

    start_time = time.time()

    # Accelerator
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    num_steps = args.epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_steps))

    model.train()

    for epoch in range(args.epochs):
        train_loss = 0
        val_loss = 0
        tr_examples, tr_steps = 0, 0
        val_examples, val_steps = 0, 0

        watcher = ww.WeightWatcher(model=model)
        ww_details = watcher.analyze(min_evals=0)
        ww_details.to_csv(os.path.join(stats_path, f"epoch_{epoch}.csv"))

        print(f"=======>Epoch {epoch+1}/{args.epochs}")

        if epoch == 0:  # Done
            # CHOOSING LAYERS TO TRAIN
            filtered = ww_details[
                ww_details["longname"].str.contains("nw_layer|classifier")
            ]
            train_names = (
                filtered.sort_values(by=["alpha"], ascending=args.alpha_ascending)[
                    "longname"
                ]
                .iloc[: args.num_layers]
                .to_list()
            )
            print("Training layers:", train_names)
            layer_to_train = []
            for layer in train_names:
                layer_to_train.append(layer + ".weight")
                layer_to_train.append(layer + ".bias")
                if args.add_layer_norm:
                    if "output" in layer:
                        layer_to_train.append(
                            layer.replace("dense", "LayerNorm") + ".weight"
                        )
                        layer_to_train.append(
                            layer.replace("dense", "LayerNorm") + ".bias"
                        )
            layer_to_train = list(set(layer_to_train))
            print("Training layers:", layer_to_train)
            for name, param in model.named_parameters():
                if name in layer_to_train:
                    print(f"Enabling {name} parameter")
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            # batch = tuple(k.to(device) for k in batch)
            outputs = model(
                **batch,
                # token_type_ids=None,
                # attention_mask=batch[1].to(device),
                # labels=batch[2].to(device),
            )
            # output.loss.backward()
            accelerator.backward(outputs.loss)
            optimizer.step()
            progress_bar.update(1)
            train_loss += outputs.loss.item()
            tr_examples += len(batch[0])
            num_all_pts += len(batch[0])
            tr_steps += 1
            train_losses.append(train_loss / tr_steps)

            if step == 0:
                freeze_dict = defaultdict(list)
                for name, param in model.named_parameters():
                    freeze_dict["name"].append(name)
                    if param.grad is None:
                        freeze_dict["freeze_layer"].append(True)
                    elif torch.sum(param.grad.abs()).item() > 0:
                        freeze_dict["freeze_layer"].append(False)
                pd.DataFrame(freeze_dict).to_csv(
                    os.path.join(stats_path, f"freeze_{epoch}.csv")
                )
            time_elapsed = (time.time() - start_time) / 60

            val_loss, val_acc = calc_val_loss(model, eval_dataloader, device)
            print(
                f"Epoch: {epoch+1}/{args.epochs} | Time Elapsed: {time_elapsed:.2f} mins | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )
            val_losses.append(val_loss)
            val_accs.append(val_acc)

    return train_losses, val_losses, val_accs





if args.task_name is not None:
    raw_datasets = load_dataset("glue", args.task_name)
    is_regression = args.task_name == "stsb"
    if is_regression:
        num_labels = 1
    else:
        label_list = raw_datasets["train"].features["label"].names  # type: ignore
        num_labels = len(label_list)
else:
    raw_datasets = load_dataset("glue", "all")
    is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]  # type: ignore
    if is_regression:
        num_labels = 1
    else:
        label_list = raw_datasets["train"].unique("label")  # type: ignore
        label_list.sort()
        num_labels = len(label_list)






model = get_model(args, accelerator.device)
optimizer = getOptim(model, True, 1)
tokenizer = AutoTokenizer.from_pretrained(
    args.model_name,
    use_fast=not args.slow_tokenizer,
)

no_decay = ["bias", "LayerNorm.weight"]


def get_model_params(model):  # WORK IN PROGRESS
    params = {}
    for name in model.state_dict():
        params[name] = copy.deepcopy(model.state_dict()[name])
    return params


lr_scheduler = get_scheduler(  # WORK IN PROGRESS
    name=SchedulerType.LINEAR,
    optimizer=optimizer,
    num_warmup_steps=args.warmup_steps,
    num_training_steps=args.max_train_steps,
)

if args.task_name is not None:
    sentence1_key, sentence2_key = task_to_keys[args.task_name]
else:
    sentence1_key, sentence2_key = "sentence1", "sentence2"

padding = "max_length" if args.pad_to_max_length else False


def get_train_eval():  # WORK IN PROGRESS
    def preprocess(examples):
        texts = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *texts, padding=padding, max_length=args.max_length, truncation=True
        )

        return result

    processed_datasets = raw_datasets.map(
        preprocess,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,  # type: ignore
        # desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]  # type: ignore
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]  # type: ignore

    for index in random.sample(range(len(train_dataset)), 3):
        print(f"Sample {index} of train set: {train_dataset[index]}")

    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    train_dataloader = DataLoader(
        train_dataset,  # type: ignore
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.batch_size,
    )

    eval_dataloader = DataLoader(
        eval_dataset,  # type: ignore
        collate_fn=data_collator,
        batch_size=args.batch_size,
    )

    return train_dataloader, eval_dataloader


train_dataloader, eval_dataloader = get_train_eval()

overrode_max_train_steps = False
num_steps_per_epoch = math.ceil(
    len(train_dataloader) / args.gradient_accumulation_steps
)

if args.max_train_steps is None:
    args.max_train_steps = args.epochs * num_steps_per_epoch
    overrode_max_train_steps = True


if args.task_name is None:
    metric = evaluate.load(
        "glue",
        args.task_name,
        experiment_id=str(args.seed) + str(args.num_layers) + str(args.alpha_ascending),
    )
else:
    metric = evaluate.load("accuracy")

# train stuff
progress_bar = tqdm(
    range(args.max_train_steps), disable=not accelerator.is_local_main_process
)


# def evaluate(model, eval_dataloader):

#     all_acc=[]
#     all_loss=[]
#     ##### evaluate #####
#     print ("begin evaluate ======================")
#     model.eval()
#     samples_seen = 0
#     total_loss=0
#     for step, batch in enumerate(eval_dataloader):
#         with torch.no_grad():
#             outputs = model(**batch)
#             loss = outputs.loss
#             total_loss += loss.detach().float()
#         predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
#         predictions, references = accelerator.gather((predictions, batch["labels"]))
#         # If we are in a multiprocess environment, the last batch has duplicates
#         if accelerator.num_processes > 1:
#             if step == len(eval_dataloader) - 1:
#                 predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
#                 references = references[: len(eval_dataloader.dataset) - samples_seen]
#             else:
#                 samples_seen += references.shape[0]
#         metric.add_batch(
#             predictions=predictions,
#             references=references,
#         )
#     eval_metric = metric.compute()
#     if args.task_name == "mnli":
#         total_loss=0
#         # Final evaluation on mismatched validation set
#         eval_dataset = processed_datasets["validation_mismatched"]
#         eval_dataloader = DataLoader(
#             eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
#         )
#         eval_dataloader = accelerator.prepare(eval_dataloader)
#         model.eval()
#         for step, batch in enumerate(eval_dataloader):
#             outputs = model(**batch)

#             loss = outputs.loss
#             total_loss += loss.detach().float()
#             predictions = outputs.logits.argmax(dim=-1)
#             metric.add_batch(
#                 predictions=accelerator.gather(predictions),
#                 references=accelerator.gather(batch["labels"]),
#             )
#         eval_metric = metric.compute()

#     all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
#     print    (all_results)
#     # total_loss.item() / len(eval_dataloader)
#     print ("loss",total_loss.item() / len(eval_dataloader))
