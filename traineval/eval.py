import torch


# Validation Loss (Classification)
def calc_val_loss(args, model, eval_dataloader, device, accelerator=None):  # Done
    """
    Args:
        args: A dictionary of arguments
        model: A model object
        eval_dataloader: A dataloader for evaluation
        device: A string of device name

    Returns:
        loss: A float of validation loss
        acc: A float of validation accuracy
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
