import itertools

from numpy import sort
from sklearn.neighbors import sort_graph_by_row_values
from gputracker.gputracker import get_logger, DispatchThread

gpus = list(range(8))
# gpus  = [5, 6, 7]
train_layers = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 18, 24, 30, 36, 72, 74]
task_list = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb']

model = "bert-base-uncased"
ascending_order = ["True", "False"]
sortby = ["alpha", "layer"]

norm = "False"
freeze_bert = "True"
train_seed_lst = [5, 6, 7]
seed = 42
max_length = 128
batch_size = 32
epochs = 3
logger = get_logger('log', 'schedule_subspace.log')

grid = list(
    itertools.product(
        task_list,
        sortby,
        # ascending_order, 
        train_layers,)
)
BASH_COMMAND_LIST = []
task = "cola"
sby = "alpha"
order  = "False"

for task, sby, layers in grid:
    
    save_path = "/rscratch/tpang/kinshuk/RpMKin/bert_ft/GLUE" + \
        f"/trainseed_{seed}/task_{task}/lay_norm_{norm}/{sby}_asc_{order}/layers_{layers}/lr2e-5_epoch3_bs{batch_size}/"

    cmd = "OMP_NUM_THREADS=1 python /rscratch/tpang/kinshuk/RpMKin/bert_ft/bertft.py " + \
        f"--savepath {save_path} " + \
        f"--epochs {epochs} " + \
        f"--model_name {model} " + \
        f"--task_name {task} " + \
        f"--sortby {sby} " + \
        f"--alpha_ascending {order} " + \
        f"--batch_size {batch_size} " + \
        f"--learning_rate 2e-5 " + \
        f"--seed {seed} " + \
        f"--num_layers {layers} " + \
        f"--verbose False " + \
        f"--debug False " + \
        f"--memlog False"
        
    BASH_COMMAND_LIST.append(cmd)


dispatch_thread = DispatchThread(
    "GLUE dataset training",
    BASH_COMMAND_LIST,
    logger,
    gpu_m_th=50,
    gpu_list=gpus,
    maxcheck=0,
)

dispatch_thread.start()
dispatch_thread.join()

import time
time.sleep(5)

logger.info("Exiting Main Thread")