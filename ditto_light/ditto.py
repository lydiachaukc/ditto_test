import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import sklearn.metrics as metrics
import argparse

from .dataset import DittoDataset
from torch.utils import data
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, WeightedRandomSampler, SequentialSampler, TensorDataset, RandomSampler
from tensorboardX import SummaryWriter
from apex import amp

import pandas as pd
from ditto_light.classification_NN import classification_NN
from torch.nn import CosineSimilarity

lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased'}

class DittoModel(nn.Module):
    """A baseline model for EM."""

    def __init__(self, device='cuda', lm='roberta', alpha_aug=0.8, num_hidden_lyr=1, num_input_dimension=1):
        super().__init__()
        if lm in lm_mp:
            self.bert = AutoModel.from_pretrained(lm_mp[lm])
        else:
            self.bert = AutoModel.from_pretrained(lm)

        self.device = device
        self.alpha_aug = alpha_aug

        # linear layer
        hidden_size = self.bert.config.hidden_size
        self.classifier = torch.nn.Linear((num_input_dimension + hidden_size), 2)
        
        #---new
        if (num_input_dimension != 1):
            cos = CosineSimilarity()
            self.calculate_similiarity = lambda a, b: cos(a,b).view(-1,1)
        else:
            self.calculate_similiarity = self.calculate_difference
        
        # self.classifier = classification_NN(
        #     #inputs_dimension = 1 + hidden_size,
        #     num_hidden_lyr = num_hidden_lyr,
        #     dropout_prob = 0.2)

    def forward(self, x1, attention_mask, token_type_ids, num1, num2, 
                x2=None, attention_mask_aug=None, token_type_id_aug=None):
        """Encode the left, right, and the concatenation of left+right.

        Args:
            x1 (LongTensor): a batch of ID's
            x2 (LongTensor, optional): a batch of ID's (augmented)

        Returns:
            Tensor: binary prediction
        """
        x1 = x1.to(self.device) # (batch_size, seq_len)
        attention_mask = attention_mask.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        
        num1 = num1.to(self.device)
        num2 = num2.to(self.device)
        # calculate cossine similiary of numeric features
        numerical_features = self.calculate_similiarity(num1, num2)
        
        if x2 is not None:
            # MixDA
            x2 = x2.to(self.device) # (batch_size, seq_len)
            attention_mask_aug = attention_mask_aug.to(self.device)
            token_type_id_aug = token_type_id_aug.to(self.device)
            enc = self.bert(input_ids =torch.cat((x1, x2)),
                            attention_mask  = torch.cat((attention_mask, attention_mask_aug)),
                            token_type_ids = torch.cat((token_type_ids, token_type_id_aug))
                            )[0][:, 0, :]
            batch_size = len(x1)
            enc1 = enc[:batch_size] # (batch_size, emb_size)
            enc2 = enc[batch_size:] # (batch_size, emb_size)

            aug_lam = np.random.beta(self.alpha_aug, self.alpha_aug)
            enc = enc1 * aug_lam + enc2 * (1.0 - aug_lam)
        else:
            enc = self.bert(input_ids = x1,
                            attention_mask  = attention_mask,
                            token_type_ids = token_type_ids
                            )[0][:, 0, :]
            
        return self.classifier(torch.cat((enc, numerical_features.view(-1,1)), dim=1))
    
    def calculate_difference(self, tensorA, tensorB):
        return torch.nan_to_num(torch.abs(tensorA - tensorB) *2 / (tensorA + tensorB))


def evaluate(model, iterator, threshold=None):
    """Evaluate a model on a validation/test dataset

    Args:
        model (DMModel): the EM model
        iterator (Iterator): the valid/test dataset iterator
        threshold (float, optional): the threshold on the 0-class

    Returns:
        float: the F1 score
        float (optional): if threshold is not provided, the threshold
            value that gives the optimal F1
    """
    all_p = []
    all_y = []
    all_probs = []
    with torch.no_grad():
        for batch in iterator:
            x, y , attention_mask, token_type_ids, num1, num2 = batch
            logits = model(x, attention_mask, token_type_ids, num1, num2)
            probs = logits.softmax(dim=1)[:, 1]
            all_probs += probs.cpu().numpy().tolist()
            all_y += y.cpu().numpy().tolist()

    if threshold is not None:
        pred = [1 if p > threshold else 0 for p in all_probs]
        f1 = metrics.f1_score(all_y, pred)
        return f1
    else:
        best_th = 0.5
        f1 = 0.0 # metrics.f1_score(all_y, all_p)

        for th in np.arange(0.0, 1.0, 0.05):
            pred = [1 if p > th else 0 for p in all_probs]
            new_f1 = metrics.f1_score(all_y, pred)
            if new_f1 > f1:
                f1 = new_f1
                best_th = th

        return f1, best_th


def train_step(train_iter, model, optimizer, scheduler, hp):
    """Perform a single training step

    Args:
        train_iter (Iterator): the train data loader
        model (DMModel): the model
        optimizer (Optimizer): the optimizer (Adam or AdamW)
        scheduler (LRScheduler): learning rate scheduler
        hp (Namespace): other hyper-parameters (e.g., fp16)

    Returns:
        None
    """
    criterion = nn.CrossEntropyLoss()
    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()

        if len(batch) == 6:
            x, y, attention_mask, token_type_ids, num1, num2 = batch
            prediction = model(x, attention_mask, token_type_ids, num1, num2)
        else:
            x, y, attention_mask, token_type_ids, num1, num2, x_aug, attention_mask_aug, token_type_ids_aug = batch
            prediction = model(x, attention_mask, token_type_ids, num1, num2, x_aug, attention_mask_aug, token_type_ids_aug)

        loss = criterion(prediction, y.to(model.device))

        if hp.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 10 == 0: # monitoring
            print(f"step: {i}, loss: {loss.item()}")
        del loss


def train(trainset, validset, testset, run_tag, hp):
    """Train and evaluate the model

    Args:
        trainset (DittoDataset): the training set
        validset (DittoDataset): the validation set
        testset (DittoDataset): the test set
        run_tag (str): the tag of the run
        hp (Namespace): Hyper-parameters (e.g., batch_size,
                        learning rate, fp16)

    Returns:
        None
    """
    padder = trainset.pad
    # create the DataLoaders
    train_iter = prepare_data_loader(trainset, hp.batch_size, padder)
    valid_iter = data.DataLoader(dataset=validset,
                                 batch_size=hp.batch_size*16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)
    test_iter = data.DataLoader(dataset=testset,
                                 batch_size=hp.batch_size*16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)

    # initialize model, optimizer, and LR scheduler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DittoModel(device=device,
                       lm=hp.lm,
                       alpha_aug=hp.alpha_aug,
                       num_input_dimension=len(trainset.num_pairs[0][0]))
    model = model.cuda()
    optimizer = AdamW(model.parameters(), lr=hp.lr)

    if hp.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    num_steps = (len(trainset) // hp.batch_size) * hp.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)

    # logging with tensorboardX
    writer = SummaryWriter(log_dir=hp.logdir)
    output = pd.read_csv(hp.logdir + "result.csv")

    best_dev_f1 = best_test_f1 = 0.0
    for epoch in range(1, hp.n_epochs+1):
        # train
        model.train()
        train_step(train_iter, model, optimizer, scheduler, hp)

        # eval
        model.eval()
        dev_f1, th = evaluate(model, valid_iter)
        test_f1 = evaluate(model, test_iter, threshold=th)

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_test_f1 = test_f1
            if hp.save_model:
                # create the directory if not exist
                directory = os.path.join(hp.logdir, hp.task)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # save the checkpoints for each component
                ckpt_path = os.path.join(hp.logdir, hp.task, 'model.pt')
                ckpt = {'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch}
                torch.save(ckpt, ckpt_path)

        print(f"epoch {epoch}: dev_f1={dev_f1}, f1={test_f1}, best_f1={best_test_f1}")
        output = output.append({"Tag": run_tag, "Data": hp.task, "Epochs": epoch, "dev_f1": dev_f1, "test_f1": test_f1, "Model": "numd"},
                               ignore_index=True)

        # logging
        scalars = {'f1': dev_f1,
                   't_f1': test_f1}
        writer.add_scalars(run_tag, scalars, epoch)

    writer.close()
    output.to_csv(hp.logdir + "result.csv", index = False)
    
def prepare_data_loader(dataset, batch_size, padder, wighted = True):

    if not wighted:
        return DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0,
                            collate_fn=padder,
                            drop_last = True)
     
    # handle unbalanced data
    positive = sum(dataset.labels)
    counts = len(dataset.labels)
    negative = counts - positive
    weights = [lab / positive + (1-lab) / negative for lab in dataset.labels]
    
    
    return DataLoader(dataset=dataset,
                    sampler = WeightedRandomSampler(weights, counts),
                    batch_size = batch_size,
                    num_workers=0,
                    collate_fn=padder,
                    drop_last = True
                )
