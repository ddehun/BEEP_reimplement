import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
from utils.config import get_parser
from utils.utils import set_seed, dump_config, setup_path
from utils.datasets import get_trec_examples, TRECDataset, biencoder_collate_fn, split_train_valid_test
import torch
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from functools import partial
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
from transformers import AutoModel, AutoTokenizer, get_scheduler, logging


def main(args):
    set_seed(args.seed)
    setup_path(args.retriever_exp_path)
    dump_config(args, args.retriever_exp_path)
    tokenizer = AutoTokenizer.from_pretrained(args.biencoder_lm_ckpt)

    """
    Load TREC2016 CDS datasets
    """
    if not os.path.exists(args.retriever_ids_pck_path.format("train")) or not os.path.exists(args.retriever_ids_pck_path.format("valid")):
        trec_examples, queries, docs = get_trec_examples()
        train_examples, valid_examples, test_examples = split_train_valid_test(trec_examples)
    else:
        train_examples, valid_examples, queries, docs = [None] * 4
    train_dataset = TRECDataset(tokenizer, train_examples, queries, docs, args.retriever_ids_pck_path.format("train"))
    valid_dataset = TRECDataset(tokenizer, valid_examples, queries, docs, args.retriever_ids_pck_path.format("valid"))

    """
    Dataloaders
    """
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=partial(biencoder_collate_fn, pad_id=tokenizer.pad_token_id),
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=partial(biencoder_collate_fn, pad_id=tokenizer.pad_token_id),
        drop_last=False,
    )
    """
    Model Intialization
    """
    device = torch.device("cuda")
    model = AutoModel.from_pretrained(args.biencoder_lm_ckpt).to(device)
    scaler = GradScaler()

    """
    Train setup
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        patience=2,
        verbose=False,
    )
    num_update_steps_per_epoch = len(train_loader)

    triplet_loss = nn.TripletMarginLoss(margin=args.retrieval_margin)
    writer = SummaryWriter(os.path.join(args.retriever_exp_path, "board"))

    torch.save(
        model.state_dict(),
        os.path.join(args.retriever_exp_path, "models", "begin_model.pth"),
    )
    print(f"Train example: {len(train_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Dataloader length: {len(train_loader)}")
    global_step = 0

    best_loss = 99999
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        progress_bar = tqdm(range(num_update_steps_per_epoch))
        ########
        # Train
        ########
        model.train()
        for step, batch in enumerate(train_loader):
            ids, mask = (e.to(device, non_blocking=True) for e in batch)
            bs = ids.size(0) // 3
            with autocast():
                encoded = model(ids, mask).pooler_output.reshape(bs, 3, -1)  # 3 for query, positive_doc, negative_doc
                loss = triplet_loss(encoded[:, 0], encoded[:, 1], encoded[:, 2])
                scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            writer.add_scalar("Loss/train", loss, global_step)
            progress_bar.update(1)
            global_step += 1

        ############
        # Validation
        ############
        model.eval()
        loss_list = []
        for step, batch in enumerate(tqdm(valid_loader)):
            ids, mask = (e.to(device, non_blocking=True) for e in batch)
            bs = ids.size(0) // 3
            with torch.no_grad():
                encoded = model(ids, mask).pooler_output.reshape(bs, 3, -1)  # 4 for query, positive_doc, negative_doc
                loss = triplet_loss(encoded[:, 0], encoded[:, 1], encoded[:, 2]).cpu().numpy()
                loss_list.append(loss)
        valid_loss = np.mean(loss_list)
        lr_scheduler.step(valid_loss)
        writer.add_scalar("Loss/valid", valid_loss, global_step)

        torch.save(
            model.state_dict(),
            os.path.join(args.retriever_exp_path, "models", f"epoch{epoch}.pth"),
        )

        if valid_loss < best_loss:
            print("New best model")
            best_loss = valid_loss
            torch.save(
                model.state_dict(),
                os.path.join(args.retriever_exp_path, "models", f"best_model.pth"),
            )

        progress_bar.close()


if __name__ == "__main__":
    args = get_parser()
    main(args)
