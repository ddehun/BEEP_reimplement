import os
import random
import sys
from typing import Dict, List, Tuple, Union

import ir_datasets
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
from functools import partial

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from utils.utils import read_pickle

"""
Interface with (raw) datasets
"""


def get_trec_examples():
    print("[*] Building TREC examples")
    trecdataset = ir_datasets.load("pmc/v2/trec-cds-2016")

    examples = {}
    for example in trecdataset.qrels_iter():
        qid, docid, relevance = example.query_id, example.doc_id, example.relevance
        if qid not in examples:
            examples[qid] = {}
        examples[qid][docid] = True if relevance >= 1 else False
    docs = {}
    for doc in trecdataset.docs_iter():
        docid, abstract = doc.doc_id, doc.abstract
        docs[docid] = abstract
    queries = {}
    for query in trecdataset.queries_iter():
        qid, ttype, note = query.query_id, query.type, query.note
        queries[qid] = f"What is the {ttype}? {note}"

    total_examples = []
    # Make query-level examples
    for qid, articles in examples.items():
        positive_list, negative_list = [], []
        for article_id, relevance in articles.items():
            if relevance:
                positive_list.append(article_id)
            else:
                negative_list.append(article_id)
        # Hard-negative strategy following the original BEEP paper
        negative_list = negative_list[: len(positive_list)]
        total_examples += [[qid, pos, neg] for pos, neg in zip(positive_list, negative_list)]
    return total_examples, queries, docs


def get_mimic_dataset(setname, fname_template, task) -> List[Dict[str, Union[int, str]]]:
    assert setname in ["train", "valid", "test"]
    setname = "val" if setname == "valid" else setname
    fname = fname_template.format(task, setname)
    df = pd.read_csv(fname).rename(columns={"hospital_expire_flag": "label", "los_label": "label"})
    return df.to_dict("records")


def split_train_valid_test(exs: List, eval_ratio: int = 0.1) -> Tuple[List]:
    random.shuffle(exs)
    train = exs[: int((1 - 2 * eval_ratio) * len(exs))]
    valid = exs[int((1 - 2 * eval_ratio) * len(exs)) : int((1 - eval_ratio) * len(exs))]
    test = exs[int((1 - eval_ratio) * len(exs)) :]
    return train, valid, test


"""
Collator functions for dataloader
"""


def augmented_predictor_collate_fn(examples, pad_id):
    # {"id": id_, "mimic_input_ids": mimic_f, "pubmed_input_ids_list": doc_f, "pubmed_scores": doc_p, "label": label}

    # mimic example
    all_ids = [torch.tensor(ex["mimic_input_ids"]) for ex in examples]
    mask_list = list(map(partial(torch.ones, dtype=torch.long), list(map(len, all_ids))))

    # pubmed articles
    pubmed_articles_list = list(map(lambda x: x["pubmed_input_ids_list"], examples))
    bs, k = len(pubmed_articles_list), len(pubmed_articles_list[0])
    all_pubmeds_ids = [torch.tensor(ids) for articles in pubmed_articles_list for ids in articles]
    all_pubmeds_mask = list(map(partial(torch.ones, dtype=torch.long), list(map(len, all_pubmeds_ids))))
    assert len(all_pubmeds_ids) == bs * k

    # labels and ids
    example_id_list = torch.tensor([ex["id"] for ex in examples])
    labels = torch.tensor([ex["label"] for ex in examples], dtype=torch.long)
    all_ids = pad_sequence(all_ids, batch_first=True, padding_value=pad_id)
    mask = pad_sequence(mask_list, batch_first=True, padding_value=0)
    pubmed_scores = torch.tensor([e["pubmed_scores"] for e in examples])

    all_pubmeds_ids = pad_sequence(all_pubmeds_ids, batch_first=True, padding_value=pad_id)
    all_pubmeds_mask = pad_sequence(all_pubmeds_mask, batch_first=True, padding_value=0)

    return (
        all_ids,
        mask,
        all_pubmeds_ids,
        all_pubmeds_mask,
        pubmed_scores,
        labels,
        example_id_list,
    )


def predictor_collate_fn(examples, pad_id, return_example_id: bool = False):
    example_id_list = torch.tensor([ex["id"] for ex in examples])
    all_ids = [torch.tensor(ex["input_ids"]) for ex in examples]
    mask_list = list(map(partial(torch.ones, dtype=torch.long), list(map(len, all_ids))))
    labels = torch.tensor([ex["label"] for ex in examples], dtype=torch.long)
    all_ids = pad_sequence(all_ids, batch_first=True, padding_value=pad_id)
    mask = pad_sequence(mask_list, batch_first=True, padding_value=0)

    return all_ids, mask, labels, example_id_list


def biencoder_collate_fn(examples, pad_id):
    all_ids = [ids for ex in examples for ids in ex]
    mask_list = list(map(partial(torch.ones, dtype=torch.long), list(map(len, all_ids))))
    all_ids = pad_sequence([torch.tensor(seq) for seq in all_ids], batch_first=True, padding_value=pad_id)
    mask = pad_sequence(mask_list, batch_first=True, padding_value=0)
    return all_ids, mask


def reranker_collate_fn(examples, pad_id):
    all_ids = [torch.tensor(ex["input_ids"]) for ex in examples]
    mask_list = list(map(partial(torch.ones, dtype=torch.long), list(map(len, all_ids))))
    labels = torch.tensor([ex["label"] for ex in examples], dtype=torch.long)
    all_ids = pad_sequence(all_ids, batch_first=True, padding_value=pad_id)
    mask = pad_sequence(mask_list, batch_first=True, padding_value=0)
    return all_ids, mask, labels


"""
Dataset classes
"""


class RetrievalAugmentedMIMICDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        mimic_examples: List[Dict],
        pubmed_rank_results_dir: str,
        pubmed_examples_fname: str,
        pickled_fname,
        k: int,
        max_seq_len: int = 512,
    ):
        self.k = k
        self.tokenizer = tokenizer
        self.pickled_fname = pickled_fname
        self.max_length = max_seq_len
        self.features = self._featurize(mimic_examples, pubmed_rank_results_dir, pubmed_examples_fname)

    def __getitem__(self, idx):
        return self.features[idx]

    def __len__(self):
        return len(self.features)

    def _read_pubmed_results(self, path) -> Dict[str, List[Tuple[str, float]]]:
        print("[*] Read pubmed ranking results")
        flist = os.listdir(path)
        print(f"  [*] {len(flist)} files")
        results = {}
        for fname in flist:
            mimic_example_id = fname.split(".")[-2]
            with open(os.path.join(path, fname), "rb") as f:
                data = pickle.load(f)[: self.k]
            results[mimic_example_id] = data
        return results

    def _featurize(self, mimic_examples, pubmed_rank_results_dir, pubmed_examples_fname):
        if os.path.exists(self.pickled_fname):
            os.makedirs(os.path.dirname(self.pickled_fname), exist_ok=True)
            with open(self.pickled_fname, "rb") as f:
                return pickle.load(f)

        pubmed_rank_results = self._read_pubmed_results(pubmed_rank_results_dir)
        pubmed_examples = read_pickle(pubmed_examples_fname)

        ids = []
        mimic_features = []
        docs_features = []
        doc_scores = []
        labels = []

        for idx, ex in enumerate(mimic_examples):
            id_, text, label = ex["id"], ex["text"], ex["label"]
            pubmed_article_ids = list(map(lambda x: x[0], pubmed_rank_results[str(id_)]))
            pubmed_article_scores = list(map(lambda x: x[1], pubmed_rank_results[str(id_)]))
            pubmed_articles = list(map(lambda doc_id: pubmed_examples[doc_id]["text"], pubmed_article_ids))

            mimic_features.append(
                self.tokenizer(
                    text,
                    padding=False,
                    truncation=True,
                    max_length=self.max_length,
                )["input_ids"]
            )
            docs_features.append(
                self.tokenizer(
                    pubmed_articles,
                    padding=False,
                    truncation=True,
                    max_length=self.max_length,
                )["input_ids"]
            )
            ids.append(id_)
            labels.append(label)
            doc_scores.append(pubmed_article_scores)

            assert len(ids) == len(mimic_features) == len(docs_features) == len(doc_scores) == len(labels)

        features = [
            {
                "id": id_,
                "mimic_input_ids": mimic_f,
                "pubmed_input_ids_list": doc_f,
                "pubmed_scores": doc_p,
                "label": label,
            }
            for id_, mimic_f, doc_f, doc_p, label in zip(ids, mimic_features, docs_features, doc_scores, labels)
        ]

        with open(self.pickled_fname, "wb") as f:
            pickle.dump(features, f)
        return features


class MIMICDataset(Dataset):
    def __init__(self, tokenizer, examples, pickled_fname, max_seq_len):
        self.tokenizer = tokenizer
        self.pickled_fname = pickled_fname
        self.max_length = max_seq_len
        self.features = self._featurize(examples)

    def __getitem__(self, idx):
        return self.features[idx]

    def __len__(self):
        return len(self.features)

    def _featurize(self, examples):
        if os.path.exists(self.pickled_fname):
            os.makedirs(os.path.dirname(self.pickled_fname), exist_ok=True)
            with open(self.pickled_fname, "rb") as f:
                return pickle.load(f)

        ids = []
        features = []
        labels = []
        for ex in examples:
            id_, text, label = ex["id"], ex["text"], ex["label"]
            features.append(
                self.tokenizer(
                    text,
                    padding=False,
                    truncation=True,
                    max_length=self.max_length,
                )["input_ids"]
            )
            ids.append(id_)
            labels.append(label)
        assert len(features) == len(labels) == len(ids)
        features = [{"id": id_, "input_ids": f, "label": label} for id_, f, label in zip(ids, features, labels)]

        with open(self.pickled_fname, "wb") as f:
            pickle.dump(features, f)
        return features


class TRECDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        examples,
        queries,
        docs,
        feature_save_fname,
        max_length: int = 512,
        is_reranker: bool = False,
    ):
        self.tokenizer = tokenizer

        os.makedirs(os.path.dirname(feature_save_fname), exist_ok=True)
        if not os.path.exists(feature_save_fname):
            if not is_reranker:
                self.features = self._featurize_for_biencoder(examples, queries, docs, max_length)
            else:
                self.features = self._featurize_for_reranker(examples, queries, docs, max_length)
            with open(feature_save_fname, "wb") as f:
                pickle.dump(self.features, f)
        else:
            with open(feature_save_fname, "rb") as f:
                self.features = pickle.load(f)

    def __getitem__(self, idx):
        return self.features[idx]

    def __len__(self):
        return len(self.features)

    def _featurize_for_reranker(self, examples, queries, docs, max_length: int):
        features = []
        labels = []
        for ex in examples:
            qid, pos_doc_id, neg_doc_id = ex
            qtext = queries[qid]
            postext, negtext = docs[pos_doc_id], docs[neg_doc_id]
            features.extend(
                self.tokenizer(
                    [qtext] * 2,
                    [postext, negtext],
                    padding=False,
                    truncation=True,
                    max_length=max_length,
                )["input_ids"]
            )
            labels.extend([1, 0])
        assert len(features) == len(labels)
        features = [{"input_ids": f, "label": label} for f, label in zip(features, labels)]
        return features

    def _featurize_for_biencoder(self, examples, queries, docs, max_length: int):
        features = []
        for ex in examples:
            qid, pos_doc_id, neg_doc_id = ex
            qtext = queries[qid]
            postext, negtext = docs[pos_doc_id], docs[neg_doc_id]
            features.append(
                self.tokenizer(
                    [qtext, postext, negtext],
                    padding=False,
                    truncation=True,
                    max_length=max_length,
                )["input_ids"]
            )
        return features
