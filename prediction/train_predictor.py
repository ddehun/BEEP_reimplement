import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
from functools import partial

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from prediction.models import RetrievalAugmentedPredictor
from utils.config import get_parser
from utils.datasets import MIMICDataset, RetrievalAugmentedMIMICDataset, augmented_predictor_collate_fn, get_mimic_dataset, predictor_collate_fn
from utils.utils import dump_config, set_seed, setup_path


def main(args):
    set_seed(args.seed)
    setup_path(args.predictor_exp_path)
    dump_config(args, args.predictor_exp_path)
    tokenizer = AutoTokenizer.from_pretrained(args.biencoder_lm_ckpt)

    """
    Load MIMIC datasets
    """
    pickled_train_fname = args.predictor_ids_pck_path.format(args.task, args.num_doc_for_augment, "train")
    pickled_valid_fname = args.predictor_ids_pck_path.format(args.task, args.num_doc_for_augment, "valid")
    train_examples, valid_examples = list(map(partial(get_mimic_dataset, fname_template=args.mimic_fname, task=args.task), ["train", "valid"]))
    if args.num_doc_for_augment == 0:
        train_dataset = MIMICDataset(tokenizer, train_examples, pickled_train_fname, args.max_length)
        valid_dataset = MIMICDataset(tokenizer, valid_examples, pickled_valid_fname, args.max_length)
    else:
        train_dataset = RetrievalAugmentedMIMICDataset(
            tokenizer,
            train_examples,
            os.path.dirname(args.biencoder_retrieved_abstract_pck_path).format(args.casual_task_name),
            args.retrieved_abstract_fname.format(args.casual_task_name),
            pickled_train_fname,
            args.num_doc_for_augment,
            args.max_length,
        )
        valid_dataset = RetrievalAugmentedMIMICDataset(
            tokenizer,
            valid_examples,
            os.path.dirname(args.biencoder_retrieved_abstract_pck_path).format(args.casual_task_name),
            args.retrieved_abstract_fname.format(args.casual_task_name),
            pickled_valid_fname,
            args.num_doc_for_augment,
            args.max_length,
        )

    """
    Training
    """
    collate_fn = predictor_collate_fn if args.num_doc_for_augment == 0 else augmented_predictor_collate_fn
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=partial(collate_fn, pad_id=tokenizer.pad_token_id),
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
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
        model = AutoModelForSequenceClassification.from_pretrained(args.predictor_lm_ckpt).to(device)
    else:
        model = RetrievalAugmentedPredictor(args.predictor_lm_ckpt, args.augment_strategy, args.num_predictor_labels).to(device)
    scaler = GradScaler()
    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        patience=2,
        verbose=False,
    )
    num_update_steps_per_epoch = len(train_loader)

    writer = SummaryWriter(os.path.join(args.predictor_exp_path, "board"))

    torch.save(
        model.state_dict(),
        os.path.join(args.predictor_exp_path, "models", "begin_model.pth"),
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
            if args.num_doc_for_augment == 0:
                ids, mask, labels, _ = (e.to(device, non_blocking=True) for e in batch)
            else:
                batch = (e.to(device, non_blocking=True) for e in batch)
            with autocast():
                if args.num_doc_for_augment == 0:
                    loss = model(ids, mask, labels=labels)[0]
                else:
                    loss, _ = model(batch)
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
            if args.num_doc_for_augment == 0:
                ids, mask, labels = (e.to(device, non_blocking=True) for e in batch)
            else:
                batch = (e.to(device, non_blocking=True) for e in batch)
            with autocast():
                if args.num_doc_for_augment == 0:
                    loss = model(ids, mask, labels=labels)[0]
                else:
                    loss, _ = model(batch)
                loss_list.append(loss.cpu().numpy())
        valid_loss = np.mean(loss_list)
        lr_scheduler.step(valid_loss)
        writer.add_scalar("Loss/valid", valid_loss, global_step)

        torch.save(
            model.state_dict(),
            os.path.join(args.predictor_exp_path, "models", f"epoch{epoch}.pth"),
        )

        if valid_loss < best_loss:
            print("New best model")
            best_loss = valid_loss
            torch.save(
                model.state_dict(),
                os.path.join(args.predictor_exp_path, "models", f"best_model.pth"),
            )

        progress_bar.close()


if __name__ == "__main__":
    args = get_parser()
    main(args)
