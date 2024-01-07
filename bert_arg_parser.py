import argparse

parser = argparse.ArgumentParser(description="BERT Fine-Tuning")
# Parser Arguments and Defaults
parser.add_argument(
    "--savepath",
    type=str,
    default="/scratch/vipul/models",
    help="",
)
parser.add_argument("--epochs", type=int, default=3, help="")
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
    default=False,
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