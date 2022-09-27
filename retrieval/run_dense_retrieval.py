import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
from utils.config import get_parser, MIMIC_TASKNAME_MAP
from utils.utils import set_seed, read_pickle
from utils.datasets import get_mimic_dataset, MIMICDataset, biencoder_collate_fn, split_train_valid_test
import torch
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from functools import partial
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics.pairwise import euclidean_distances


import pickle
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
from transformers import AutoModel, AutoTokenizer, get_scheduler, logging, AutoModelForSequenceClassification


def main(args):
    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.biencoder_retrieved_abstract_pck_path).format(args.casual_task_name), exist_ok=True)
    os.makedirs(os.path.dirname(args.reranker_abstract_score_pck_path).format(args.casual_task_name), exist_ok=True)
    os.makedirs(os.path.dirname(args.encoded_abstract_fname).format(args.task), exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.biencoder_lm_ckpt)

    retrieved_abstracts = read_pickle(args.retrieved_abstract_fname)
    """
    Load MIMIC3 datasets
    """
    train_examples, valid_examples, test_examples = list(
        map(partial(get_mimic_dataset, fname_template=args.mimic_fname, task=args.task), ["train", "valid", "test"])
    )
    mimic_examples = train_examples + valid_examples + test_examples
    """
    Model Intialization
    """
    device = torch.device("cuda")
    model = AutoModel.from_pretrained(args.biencoder_lm_ckpt).to(device).eval()
    model.load_state_dict(torch.load(args.retriever_exp_path + "models/best_model.pth"))

    reranker = AutoModelForSequenceClassification.from_pretrained(args.reranker_lm_ckpt).to(device).eval()
    reranker.load_state_dict(torch.load(args.reranker_exp_path + "models/best_model.pth"))

    """
    Encode retrieved abstracts using bi-encoder
    """
    if os.path.exists(args.encoded_abstract_fname.format(args.task)):
        with open(args.encoded_abstract_fname.format(args.task), "rb") as f:
            article_ids, article_matrix = pickle.load(f)
    else:
        abstracts_embed = {}
        for docid, el in tqdm(retrieved_abstracts.items()):
            text = tokenizer(el["text"], max_length=args.max_length, truncation=True, padding=True, return_tensors="pt")
            with torch.no_grad():
                text = {x: y.cuda() for x, y in text.items()}
                res = model(**text)
                abstracts_embed[docid] = res["last_hidden_state"][0, 0, :].cpu().numpy()

        article_items = list(abstracts_embed.items())
        article_ids, article_matrix = [x[0] for x in article_items], [x[1] for x in article_items]
        article_matrix = np.vstack(article_matrix)
        with open(args.encoded_abstract_fname.format(args.task), "wb") as f:
            pickle.dump([article_ids, article_matrix], f)

    """
    Encode MIMIC3 dataset
    """
    outcome_questions = {
        "mortality": "What is the hospital mortality?",
        "pmv": "What is the probability of prolonged mechanical ventilation?",
        "los": "What is the probable length of stay?",
    }

    softmax_func = torch.nn.Softmax(dim=1)
    for _, example in enumerate(tqdm(mimic_examples)):
        docid, text = example["id"], example["text"]
        query_text = outcome_questions[args.casual_task_name] + " " + text
        text = tokenizer(query_text, max_length=args.max_length, truncation=True, padding=True, return_tensors="pt")

        """
        Bi-encoder
        """
        with torch.no_grad():
            text = {x: y.cuda() for x, y in text.items()}
            res = model(**text)
            cur_query_embed = res["last_hidden_state"]
        cur_query_embed = cur_query_embed[0, 0, :].cpu().numpy().transpose()
        cur_query_embed = cur_query_embed.reshape(1, -1)
        similarities = euclidean_distances(cur_query_embed, article_matrix).tolist()[0]
        ranked_docs = list(zip(article_ids, similarities))
        ranked_docs = list(sorted(ranked_docs, key=lambda x: x[1]))

        with open(args.biencoder_retrieved_abstract_pck_path.format(args.casual_task_name, docid), "wb") as f:
            pickle.dump(ranked_docs[: args.num_first_retrieval], f)

        # """
        # Reranker
        # """
        # ranked_doc_texts = [retrieved_abstracts[docid_ans_score[0]]["text"] for docid_ans_score in ranked_docs]
        # features = tokenizer(
        #     [outcome_questions[args.casual_task_name] + " " + example["text"]] * len(ranked_doc_texts),
        #     ranked_doc_texts,
        #     padding=True,
        #     truncation=True,
        #     max_length=args.max_length,
        #     return_tensors="pt",
        # )
        # print(len(features["input_ids"]))
        # bs = 100  # TODO: fix hard coding
        # ids_list, mask_list, preds = [], [], []
        # for i, (ids, mask) in enumerate(tqdm(zip(features["input_ids"], features["attention_mask"]))):
        #     ids_list.append(ids)
        #     mask_list.append(mask)
        #     if len(ids_list) == bs:
        #         ids_list, mask_list = torch.tensor(ids_list).to(device), torch.tensor(mask_list).to(device)
        #         with torch.no_grad():
        #             outputs = softmax_func(model(ids_list, mask_list)[1].cpu()).numpy()[:, 1].tolist()
        #         assert len(outputs) == bs
        #         preds.extend(outputs)
        #         ids_list, mask_list = [], []
        # reranker_preds = list(zip([e[0] for e in ranked_docs], preds))
        # with open(args.reranker_abstract_score_pck_path.format(args.casual_task_name, docid), "wb") as f:
        #     pickle.dump(reranker_preds, f)


if __name__ == "__main__":
    args = get_parser()
    main(args)
