from turtle import forward
import torch
from torch.nn import Module
from transformers import AutoModel, AutoConfig

STRATEGY = ["avg", "wavg", "svote", "wvote"]


class RetrievalAugmentedPredictor(Module):
    def __init__(self, lmname: str, strategy: str, num_label: int = 2):
        self.encoder = AutoModel.from_pretrained(lmname)
        self.lm_config = AutoConfig.from_pretrained(lmname)
        self.strategy = strategy
        self.predictor = torch.nn.Linear(2 * self.lm_config.hidden_size, num_label)
        self.softmax = torch.nn.Softmax(dim=1)
        self.ce_criteria = torch.nn.CrossEntropyLoss()
        self.nll_criteria = torch.nn.NLLLoss()

    def forward(self, batch):
        """
        batch: all_ids, mask, all_pubmeds_ids, all_pubmeds_mask, doc_scores,labels, example_id_list
        return: Tuple[logits, loss]
        """
        note_ids, note_mask, docs_ids, docs_mask, doc_scores, labels, _ = batch

        bs = note_ids.size(0)
        k = docs_ids.size(0) / bs

        note_outputs = self.encoder(note_ids, note_mask, return_dict=True).pooler_output
        docs_outputs = self.encoder(docs_ids, docs_mask, return_dict=True).pooler_output

        if "avg" in self.strategy:
            if self.strategy == "avg":
                docs_outputs = docs_outputs.reshape(bs, k, -1).mean(1)
            elif self.strategy == "wavg":
                wsum = doc_scores.reshape(bs * k).unsqueeze(1) * docs_outputs.reshape(bs * k, -1)
                wsum = wsum.reshape(bs, k, -1).sum(1)
                docs_outputs = wsum / doc_scores.sum(1).unsqueeze(1)
            concat_repr = torch.cat([note_outputs, docs_outputs], 1)
            logits = self.predictor(concat_repr)
            loss = self.ce_criteria(logits, labels)
            return loss, logits

        elif "vote" in self.strategy:
            note_outputs = note_outputs.unsqueeze(1).repeat(1, k, 1).reshape(bs * k, -1)
            concat_repr = torch.cat([note_outputs, docs_outputs.reshape(bs * k, -1)], 1)
            logits = self.predictor(concat_repr)  # bs * k, 2
            probs = self.softmax(logits)
            if self.strategy == "svote":
                probs = probs.reshape(bs, k, -1).mean(1)
            elif self.strategy == "wvote":
                probs = doc_scores.reshape(bs * k).unsqueeze(1) * probs.reshape(bs * k, -1)
                probs = probs.reshape(bs, k, -1).sum(1) / doc_scores.sum(1).unsqueeze(1)
            loss = self.nll_criteria(probs, labels)
            return loss, probs
        else:
            raise ValueError
