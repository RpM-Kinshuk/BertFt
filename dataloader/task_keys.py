# Keys for GLUE Tasks
glue_task_to_keys = {  # Done
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

superglue_task_to_keys = {  # Done
    "boolq": ("question", "passage"),
    "cb": ("premise", "hypothesis"),
    "copa": ("premise", "choice1"),
    "multirc": ("paragraph", "question"),
    "record": ("passage", "query"),
    "rte": ("premise", "hypothesis"),
    "wic": ("sentence1", "sentence2"),
    "wsc": ("text", "span1_text"),
}

def task_keys(args):
    return glue_task_to_keys # Currently only GLUE tasks are supported
    if args.task_name in glue_task_to_keys:
        return glue_task_to_keys
    elif args.task_name in superglue_task_to_keys:
        return superglue_task_to_keys