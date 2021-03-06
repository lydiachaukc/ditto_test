import torch

from torch.utils import data
from transformers import AutoTokenizer

from .augment import Augmenter

# map lm name to huggingface's pre-trained model names
lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased'}

def get_tokenizer(lm):
    if lm in lm_mp:
        return AutoTokenizer.from_pretrained(lm_mp[lm])
    else:
        return AutoTokenizer.from_pretrained(lm)


class DittoDataset(data.Dataset):
    """EM dataset"""

    def __init__(self,
                 path,
                 max_len=256,
                 size=None,
                 lm='roberta',
                 da=None):
        self.tokenizer = get_tokenizer(lm)
        self.pairs = []
        self.num_pairs = []
        self.labels = []
        self.max_len = max_len
        self.size = size

        if isinstance(path, list):
            lines = path
        else:
            lines = open(path, encoding = "ISO-8859-1")

        for line in lines:
            s1, s2, num1, num2, label = line.strip().split('\t')
            self.pairs.append((s1, s2))
            self.labels.append(int(label))
            
            num1 = self.convert_string_to_float_tensor(num1)
            num2 = self.convert_string_to_float_tensor(num2)
            self.num_pairs.append((num1, num2))

        self.pairs = self.pairs[:size]
        self.num_pairs = self.num_pairs[:size]
        self.labels = self.labels[:size]
        self.da = da
        if da is not None:
            self.augmenter = Augmenter()
        else:
            self.augmenter = None


    def __len__(self):
        """Return the size of the dataset."""
        return len(self.pairs)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the two entities
            List of int: token ID's of the two entities augmented (if da is set)
            int: the label of the pair (0: unmatch, 1: match)
        """
        left = self.pairs[idx][0]
        right = self.pairs[idx][1]

        # left + right
        x = self.tokenizer.encode_plus(text=left,
                                    text_pair=right,
                                    max_length=self.max_len,
                                    truncation=True,
                                    return_attention_mask = True,
                                    return_token_type_ids = True)

        # augment if da is set
        if self.da is not None:
            combined = self.augmenter.augment_sent(left + ' [SEP] ' + right, self.da)
            left, right = combined.split(' [SEP] ')
            x_aug = self.tokenizer.encode_plus(text=left,
                                              text_pair=right,
                                              max_length=self.max_len,
                                              truncation=True,
                                              return_attention_mask = True,
                                              return_token_type_ids = True)
            return x["input_ids"], self.labels[idx], x["attention_mask"], x["token_type_ids"],\
                self.num_pairs[idx][0], self.num_pairs[idx][1], \
                    x_aug["input_ids"], x_aug["attention_mask"], x_aug["token_type_ids"]
        else:
            return x["input_ids"], self.labels[idx], x["attention_mask"], x["token_type_ids"], \
                self.num_pairs[idx][0], self.num_pairs[idx][1]
    
    def convert_string_to_float_tensor(self, num_str):
        return list(map(float,num_str.strip().split(" ")))

    @staticmethod
    def pad(batch):
        """Merge a list of dataset items into a train/test batch
        Args:
            batch (list of tuple): a list of dataset items

        Returns:
            LongTensor: x1 of shape (batch_size, seq_len)
            LongTensor: x2 of shape (batch_size, seq_len).
                        Elements of x1 and x2 are padded to the same length
            LongTensor: a batch of labels, (batch_size,)
        """
        if len(batch[0]) == 9:
            x1, y, attention_mask, token_type_ids, num1, num2, x_aug, attention_mask_aug, token_type_ids_aug = zip(*batch)

            maxlen = max([len(x) for x in x1+x_aug])
            x1 = [xi + [0]*(maxlen - len(xi)) for xi in x1]
            attention_mask = [xi + [0]*(maxlen - len(xi)) for xi in attention_mask]
            token_type_ids = [xi + [0]*(maxlen - len(xi)) for xi in token_type_ids]
            num1 = [torch.tensor(xi, dtype=torch.float32) for xi in num1]
            num2 = [torch.tensor(xi, dtype=torch.float32) for xi in num2]
            
            x_aug = [xi + [0]*(maxlen - len(xi)) for xi in x_aug]
            attention_mask_aug = [xi + [0]*(maxlen - len(xi)) for xi in attention_mask_aug]
            token_type_ids_aug = [xi + [0]*(maxlen - len(xi)) for xi in token_type_ids_aug]
            
            return torch.LongTensor(x1), \
                   torch.LongTensor(y), \
                   torch.LongTensor(attention_mask), \
                   torch.LongTensor(token_type_ids), \
                   torch.tensor(num1), \
                   torch.tensor(num2), \
                   torch.LongTensor(x_aug), \
                   torch.LongTensor(attention_mask_aug), \
                   torch.LongTensor(token_type_ids_aug)
        else:
            x12, y, attention_mask, token_type_ids, num1, num2 = zip(*batch)
            maxlen = max([len(x) for x in x12])
            
            x12 = [xi + [0]*(maxlen - len(xi)) for xi in x12]
            attention_mask = [xi + [0]*(maxlen - len(xi)) for xi in attention_mask]
            token_type_ids = [xi + [0]*(maxlen - len(xi)) for xi in token_type_ids]
            num1 = [torch.tensor(xi, dtype=torch.float32) for xi in num1]
            num2 = [torch.tensor(xi, dtype=torch.float32) for xi in num2]
            
            return torch.LongTensor(x12), \
                   torch.LongTensor(y), \
                   torch.LongTensor(attention_mask), \
                   torch.LongTensor(token_type_ids), \
                   torch.tensor(num1), \
                   torch.tensor(num2)
