import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
from utils.config import get_parser
from utils.utils import set_seed, dump_config
from utils.datasets import get_mimic_dataset, MIMICDataset, predictor_collate_fn
import torch
import json
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from functools import partial
from sklearn.metrics import f1_score, accuracy_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import torch
import math
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from functools import partial


def main(args):
    set_seed(args.seed)
    assert os.path.exists(args.predictor_exp_path)
    tokenizer = AutoTokenizer.from_pretrained(args.biencoder_lm_ckpt)

    """
    Load MIMIC datasets
    """
    pickled_test_fname = args.predictor_ids_pck_path.format(args.task, args.predictor_input_type, "test")
    (test_examples,) = list(map(partial(get_mimic_dataset, fname_template=args.mimic_fname, task=args.task), ["test"]))
    test_dataset = MIMICDataset(tokenizer, test_examples, pickled_test_fname, args.max_length)
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=partial(predictor_collate_fn, pad_id=tokenizer.pad_token_id, return_example_id=True),
        drop_last=False,
    )
    """
    Model Intialization
    """
    device = torch.device("cuda")
    model = AutoModelForSequenceClassification.from_pretrained(args.predictor_lm_ckpt).to(device).eval()
    model.load_state_dict(torch.load(os.path.join(args.predictor_exp_path, "models", "best_model.pth")))

    """
    Evaluation
    """
    ids_list, answer_list, logit_list = [], [], []
    for step, batch in enumerate(tqdm(test_loader)):
        ids, mask, labels, example_ids = (e.to(device, non_blocking=True) for e in batch)
        with torch.no_grad():
            logits = model(ids, mask)[0].cpu().numpy()
            logit_list.append(logits)
        answer_list.append(labels.cpu().numpy())
        ids_list.append(example_ids.cpu().numpy())

    logit_list = np.concatenate(logit_list, 0).tolist()
    answer_list = np.concatenate(answer_list).tolist()
    ids_list = np.concatenate(ids_list).tolist()
    pred_list = np.argmax(logit_list, 1).tolist()
    assert len(answer_list) == len(pred_list) == len(logit_list) == len(ids_list)
    accuracy = float(accuracy_score(pred_list, answer_list))
    macro_f1 = float(f1_score(answer_list, pred_list, average="macro"))
    micro_f1 = float(f1_score(answer_list, pred_list, average="micro"))

    output = {
        "performance": {"accuracy": accuracy, "macro-f1": macro_f1, "micro-f1": micro_f1},
        "prediction": {id_: {"logit": logit_list[idx], "label": answer_list[idx]} for idx, id_ in enumerate(ids_list)},
    }
    with open(os.path.join(args.predictor_exp_path, "result.json"), "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    args = get_parser()
    main(args)
