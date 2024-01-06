import random
from librosa import cache
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    DataCollatorWithPadding,
    default_data_collator
)
from BertFt.model.getmodel import get_model
from task_keys import task_keys

# Get GLUE Train and Eval Dataloaders
def get_model_data(args, cache_dir=None, accelerator=None):
    """
    Args:
        args: A dictionary of arguments
        cache_dir: A string of cache directory
    
    Returns:
        model: A model object
        train_dataloader: A dataloader for training
        eval_dataloader: A dataloader for evaluation
    """
    num_labels = 1
    label_list = []
    task_to_keys = task_keys(args)
    # Load Raw Data and find num_labels
    if args.task_name is not None:
        raw_datasets = load_dataset(
            "glue", args.task_name, cache_dir=cache_dir
        )
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names  # type: ignore
            num_labels = len(label_list)
    else:
        raw_datasets = load_dataset(
            "glue", "all", cache_dir=cache_dir
        )
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]  # type: ignore
        if not is_regression:
            label_list = raw_datasets["train"].unique("label")  # type: ignore
            label_list.sort()
            num_labels = len(label_list)

    # Load Tokenizer
    tokenizer = BertTokenizer.from_pretrained(  # Done
        args.model_name,
        do_lower_case=True,
        use_fast=not args.slow_tokenizer,
    )

    model = get_model(args=args, num_labels=num_labels)

    # Define keys for both inputs
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        sentence1_key, sentence2_key = "sentence1", "sentence2"

    label_to_id = None

    # Set target padding
    padding = "max_length" if args.pad_to_max_length else False

    if args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id  # type: ignore
        model.config.id2label = {id: label for label, id in label_to_id.items()}  # type: ignore
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}  # type: ignore
        model.config.id2label = {id: label for label, id in model.config.label2id.items()}  # type: ignore

    # Preprocess Data
    def preprocess(input):
        """
        Args:
            input: A dictionary of input data

        Returns:
            result: A dictionary of processed data
        """
        texts = (
            (input[sentence1_key],)
            if sentence2_key is None
            else (input[sentence1_key], input[sentence2_key])
        )
        result = tokenizer(
            *texts, padding=padding, max_length=args.max_length, truncation=True
        )
        if "label" in input:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in input["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = input["label"]
        return result

    # if args.accelerate:
    #     with accelerator.main_process_first():
    #         processed_datasets = raw_datasets.map(
    #             preprocess,
    #             batched=True,
    #             remove_columns=raw_datasets["train"].column_names,  # type: ignore
    #             # desc="Running tokenizer on dataset",
    #         )
    # else:
    processed_datasets = raw_datasets.map(
        preprocess,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,  # type: ignore
        # desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]  # type: ignore
    eval_dataset = processed_datasets[  # type: ignore
        "validation_matched" if args.task_name == "mnli" else "validation"
    ]
    if args.verbose:
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

    return model, train_dataloader, eval_dataloader
