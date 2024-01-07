from model.bertbase import BertFT
from model.roberta import RobertaFT
from peft import get_peft_model, LoraConfig, TaskType # type: ignore

def get_model(args, num_labels):  # Done
    """
    Args:
        args: A dictionary of arguments
        num_labels: Number of labels
    Returns:
        model: A model object
    """
    model = None
    if "bert" in args.model_name.lower():
        model = BertFT.from_pretrained(
            args.model_name,
            num_labels=num_labels,
            problem_type="regression"
            if args.task_name == "stsb"
            else "single_label_classification",
            # cache_dir=args.savepath,
        )
    elif "roberta" in args.model_name.lower():
        model = RobertaFT.from_pretrained(
            args.model_name,
            num_labels=num_labels,
            problem_type="regression"
            if args.task_name == "stsb"
            else "single_label_classification",
            # cache_dir=args.savepath,
        )
    
    # If freeze_bert is true, freeze pre-trained layers
    if args.freeze:
        if args.verbose:
            print(f"Freezing {args.model_name} Model")
        for name, param in model.named_parameters():  # type: ignore
            if "classifier" in name or "new_layer" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    # Else, unfreeze all layers
    else:
        if args.verbose:
            print(f"Defreezing {args.model_name} Model")
        for name, param in model.named_parameters():  # type: ignore
            param.requires_grad = True
    
    # Get the LoRA injected model
    if 'autolora' in args.sortby.lower():
        lora_config = LoraConfig(
            task_type= TaskType.SEQ_CLS, #optional
            inference_mode = False,
            r = 1,
            lora_alpha = 1,
            lora_dropout = 0.05,
            bias = "none",
            # target_modules (Union[List[str],str])
            # layers_to_transform (Union[List[int],int]) 
            #layers_pattern (str)
        )
        lora_model = get_peft_model(model, lora_config) # type: ignore
        return lora_model
    return model