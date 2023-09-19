import itertools
from gputracker.gputracker import get_logger, DispatchThread

gpus = list(range(8))
train_layers = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 18, 24, 30, 36, 72, 74, 80]
task_list = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']

model = "bert-base-uncased"
ascending_order = ["True", "False"]

norm = "False"
freeze_bert = "True"
train_seed_lst = [3, 5, 7]
seed = 7
max_length = 128
batch_size = 32
epochs = 3

task = "cola"

logger = get_logger('log', 'schedule_subspace.log')

grid = list(
    itertools.product(
        ascending_order, train_layers)
)
BASH_COMMAND_LIST = []

for order, layers in grid:
    
    save_path = "/rscratch/tpang/kinshuk/RpMKin/bert_ft" + \
        f"/task_{task}/lay_norm_{norm}/alpha_asc_{order}/layers_{layers}/lr2e-5_epoch3_bs{batch_size}/"

    cmd = "OMP_NUM_THREADS=1 python bertft.py " + \
        f"--savepath {save_path} " + \
        f"--epochs {epochs} " + \
        f"--model_name {model} " + \
        f"--task_name {task} " + \
        f"--max_length {max_length} " + \
        f"--batch_size {batch_size} " + \
        f"--learning_rate 2e-5 " + \
        f"--seed {seed} " + \
        f"--freeze_bert {freeze_bert} " + \
        f"--num_layers {layers} " + \
        f"--alpha_ascending {order} " + \
        f"--slow_tokenizer True " + \
        f"--pad_to_max_length True " + \
        f"--add_layer_norm {norm} " + \
        f"--max_train_steps 1000 " + \
        f"--grad_acc_steps 1 " + \
        f"--accelerate False " + \
        f"--debug False"
        
    BASH_COMMAND_LIST.append(cmd)

dispatch_thread = DispatchThread(
    "synthetic dataset training",
    BASH_COMMAND_LIST,
    logger,
    gpu_m_th=1000,
    gpu_list=gpus,
    maxcheck=0,
)

dispatch_thread.start()
dispatch_thread.join()

import time
time.sleep(5)

logger.info("Exiting Main Thread")