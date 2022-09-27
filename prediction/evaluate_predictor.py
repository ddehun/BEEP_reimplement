import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import math
import random
from functools import partial

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from utils.config import get_parser
from utils.datasets import MIMICDataset, get_mimic_dataset, predictor_collate_fn, augmented_predictor_collate_fn, RetrievalAugmentedMIMICDataset
from utils.utils import dump_config, set_seed
from prediction.models import RetrievalAugmentedPredictor


def main(args):
    set_seed(args.seed)
    assert os.path.exists(args.predictor_exp_path)
    tokenizer = AutoTokenizer.from_pretrained(args.biencoder_lm_ckpt)

    """
    Load MIMIC datasets
    """
    pickled_test_fname = args.predictor_ids_pck_path.format(args.task, args.num_doc_for_augment, "test")
    (test_examples,) = list(map(partial(get_mimic_dataset, fname_template=args.mimic_fname, task=args.task), ["test"]))
    if args.num_doc_for_augment == 0:
        test_dataset = MIMICDataset(tokenizer, test_examples, pickled_test_fname, args.max_length)
    else:
        test_dataset = RetrievalAugmentedMIMICDataset(
            tokenizer,
            test_examples,
            os.path.dirname(args.biencoder_retrieved_abstract_pck_path).format(args.casual_task_name),
            args.retrieved_abstract_fname.format(args.casual_task_name),
            pickled_test_fname,
            args.num_doc_for_augment,
            args.max_length,
        )
    collate_fn = predictor_collate_fn if args.num_doc_for_augment == 0 else augmented_predictor_collate_fn
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=partial(collate_fn, pad_id=tokenizer.pad_token_id),
        drop_last=False,
    )
    """
    Model Intialization
    """
    device = torch.device("cuda")
    if args.num_doc_for_augment == 0:
        model = AutoModelForSequenceClassification.from_pretrained(args.predictor_lm_ckpt).to(device).eval()
    else:
        model = RetrievalAugmentedPredictor(args.predictor_lm_ckpt, args.augment_strategy, args.num_predictor_labels).to(device).eval()
    model.load_state_dict(torch.load(os.path.join(args.predictor_exp_path, "models", "best_model.pth")))

    """
    Evaluation
    """
    ids_list, answer_list, logit_list = [], [], []
    for step, batch in enumerate(tqdm(test_loader)):
        if args.num_doc_for_augment == 0:
            ids, mask, labels, example_ids = (e.to(device, non_blocking=True) for e in batch)
        else:
            batch = (e.to(device, non_blocking=True) for e in batch)
            example_ids = batch[-1]
        with torch.no_grad():
            if args.num_doc_for_augment == 0:
                logits = model(ids, mask)[0].cpu().numpy()
            else:
                _, logits = model(batch)
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
