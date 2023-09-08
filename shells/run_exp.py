import itertools
from gputrek.gputrek import get_logger, DispatchThread

gpus = list(range(8))
train_layers = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 18, 24, 30, 36, 72, 74, 80]
task_list = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']

model = "bert-base-uncased"
ascending_order = ["False", "True"]
add_layernorm = ["False", "True"]
freeze_bert = ["False", "True"]
train_seed_lst = [3, 5, 7]
seed = train_seed_lst[1]
max_length = 512

batch_size = 32
epochs = 20

logger = get_logger('log', 'schedule_subspace.log')

grid = list(
    itertools.product(
        add_layernorm, ascending_order, train_layers, task_list)
)
BASH_COMMAND_LIST = []

for norm, order, layers, task in grid:
    
    save_path = "/rscratch/tpang/kinshuk/RpMKin/bert_ft" + \
        f"/lay_norm_{norm}/alpha_asc_{order}/layers_{layers}/task_{task}/lr2e-5_epoch{epochs}_bs{batch_size}/"

    cmd = "OMP_NUM_THREADS=1 python bertft.py " + \
        f"--savepath {save_path} " + \
        f"--seed {seed} " + \
        f"--epochs {epochs} " + \
        f"--model_name {model} " + \
        f"--add_layer_norm {norm} " + \
        f"--freeze_bert {freeze_bert[0]} " + \
        f"--alpha_ascending {order} " + \
        f"--num_layers {layers} " + \
        f"--task_name {task} " + \
        f"--batch_size {batch_size} " + \
        f"--max_length {max_length} " + \
        f"--slow_tokenizer True" + \
        f"--pad_to_max_length False " + \
        f"--max_train_steps 1000 " + \
        f"--grad_acc_steps 1 " + \
        f"--accelerate True " + \
        f"--learning_rate 2e-5 "
        
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