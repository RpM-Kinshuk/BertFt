import os
import time
import torch
import random
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import weightwatcher as ww
from collections import defaultdict
from traineval.eval import calc_val_loss
# from transformers import get_scheduler


def calc_train_loss(
    args, model, optimizer, device, train_dataloader, eval_dataloader, accelerator=None
):
    """
    Args:
        args: A dictionary of arguments
        model: A model object
        optimizer: An optimizer object
        device: A string of device name
        train_dataloader: A dataloader for training
        eval_dataloader: A dataloader for evaluation
        accelerator: A 🤗 Accelerator object

    Returns:
        train_losses: A list of training losses
        val_losses: A list of validation losses
        val_accs: A list of validation accuracies
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

        if "lora" not in args.sortby.lower():
            # Save WeightWatcher Metrics
            watcher = ww.WeightWatcher(model=model)
            ww_details = watcher.analyze(min_evals=10)

        if not args.debug and "lora" not in args.sortby.lower():
            ww_details.to_csv(os.path.join(stats_path, f"epoch_{epoch}.csv"))  # type: ignore

        # CHOOSING LAYERS TO TRAIN BASED ON WEIGHTWATCHER METRICS/SORTBY
        if epoch == 0 and "lora" not in args.sortby.lower():
            filtered = ww_details[  # type: ignore
                ww_details["longname"].str.contains("new_layer|embeddings") == False  # type: ignore
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
                    filtered.sort_values(
                        by=[sortby], 
                        ascending=args.alpha_ascending
                    )["longname"].iloc[: args.num_layers].to_list()
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
            outputs.loss.backward()
            optimizer.step()
            tr_examples += len(batch["labels"])
            num_all_pts += len(batch["labels"])
            tr_steps += 1
            train_losses.append(train_loss / tr_steps)

            if not args.debug and "lora" not in args.sortby.lower():
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
            if args.task_name == 'wnli'and step >= 0.1 * len(train_dataloader):
                break
        time_elapsed = (time.time() - start_time) / 60

        # Validation Loss
        val_loss, val_acc = calc_val_loss(args, model, eval_dataloader, device)
        if args.verbose:
            print(
                f"\nEpoch: {epoch+1}/{args.epochs}"
                + f"|Elapsed: {time_elapsed:.2f} mins"
                + f"|Val Loss: {val_loss:.4f}|Val Acc: {val_acc:.4f}"
            )
        val_losses.append(val_loss)
        val_accs.append(val_acc)

    return train_losses, val_losses, val_accs
