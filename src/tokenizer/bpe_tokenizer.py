import os
import json
import logging
from typing import Dict, Union, List, Tuple
from collections import defaultdict

import torch
import numpy as np
from tqdm import trange
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from src.base.base_tokenizer import BaseTokenizer
from src.utils import write_json


logger = logging.getLogger(__name__)


class BPETokenizer(BaseTokenizer):
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    SEP_TOKEN = "<SEP>"
    
    def __init__(self, path_to_data: List[str], tokenizer_dir: str = 'saved/tokenizer',
                 vocab_size: int = 1000):
        super().__init__(path_to_data, vocab_size)
        
        os.makedirs(tokenizer_dir, exist_ok=True)
        if not os.path.exists(f'{tokenizer_dir}/vocab.json'):
            vocab, merges = self._train()
            write_json(vocab, fname=f'{tokenizer_dir}/vocab.json')
            write_json(merges, fname=f'{tokenizer_dir}/merges.json')
        
        with open(f'{tokenizer_dir}/vocab.json', 'r') as f:
            vocab = json.load(f)
        
        with open(f'{tokenizer_dir}/merges.json', 'r') as f:
            self.merges = dict()
            merges = json.load(f)
            for key, value in merges.items():
                token1, token2 = key.split(self.SEP_TOKEN)
                self.merges[(token1, token2)] = value
        
        self.ind2token = {ind: token for ind, token in enumerate(vocab)}
        self.token2ind = {token: ind for ind, token in self.ind2token.items()}
    
    def tokenize(self, text: str) -> List[str]:
        words = self.pre_tokenize(self.normalize(text))
        splits = [[c for c in word] for word in words]
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if (split[i], split[i + 1]) in self.merges:
                    token = self.merges[(split[i], split[i + 1])]
                    split = split[:i] + [token] + split[i + 2:]
                else:
                    i += 1
            splits[idx] = split

        tokens = sum(splits, [])
        for i in range(len(tokens)):
            if tokens[i] not in self.token2ind:
                tokens[i] = self.UNK_TOKEN
        return tokens
    
    def encode(self, text: Union[str, List[str]], return_tensors: bool = True) -> Dict[str, Tensor]:
        if isinstance(text, str):
            text = [text]
        
        input_ids = []
        attention_mask = []
        for t in text:
            tokenized_text = self.tokenize(t)
            input_ids.append(torch.tensor([self.token2ind[token] for token in tokenized_text]))
            attention_mask.append(torch.ones(len(input_ids[-1])))
            
        input_ids = pad_sequence(input_ids, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        if not return_tensors:
            result['input_ids'] = input_ids.numpy().tolist()
            result['attention_mask'] = attention_mask.numpy().tolist()
        
        return result

    def decode(self, token_ids: Union[int, List[int], np.ndarray, Tensor], 
               skip_special_tokens: bool = False) -> str:
        decoded_text = []
        for token_id in token_ids:
            token = self.ind2token[token_id]
            if token == self.UNK_TOKEN and skip_special_tokens:
                continue
            decoded_text.append(token)
        
        return ' '.join(decoded_text)

    def batch_decode(self, sequences: Union[List[int], List[List[int]], np.ndarray, Tensor], 
                     skip_special_tokens: bool = False) -> List[str]:
        decoded_texts = []
        for token_ids in sequences:
            decoded_texts.append(self.decode(token_ids, skip_special_tokens))
        
        return decoded_texts
    
    def _train(self):
        """
        Trains the BPE tokenizer using texts from self.path_to_data
        """
        logger.info(f"Training BPE from {self.path_to_data}...")
        vocab, word_freqs = self._get_base_vocab()
        splits = {word: [c for c in word] for word in word_freqs.keys()}

        # pair -> word -> cntr how many times pair occured in word
        pair_to_words = defaultdict(lambda: defaultdict(int))
        for word in splits:
            for a, b in zip(splits[word][:-1], splits[word][1:]):
                pair_to_words[(a, b)][word] += 1
        
        merges = dict()
        start = len(vocab) - 1
        for i in trange(start, self.vocab_size):
            best_pair = self._get_best_pair(pair_to_words, word_freqs)
            new_token = best_pair[0] + best_pair[1]
            vocab.append(new_token)
            merges[self.SEP_TOKEN.join(best_pair)] = new_token
            self._update_stats(pair_to_words, splits, best_pair)

        return vocab, merges
    
    def _get_base_vocab(self):
        """
        Reads texts from self.path_to_data and returns unique word frequencies and base vocab
        """
        
        word_freqs = defaultdict(int)
        corpus = []
        
        for path in self.path_to_data:
            with open(path, 'r') as f:
                corpus += [self.normalize(text) for text in f]

        for text in corpus:
            words = self.pre_tokenize(text)
            for word in words:
                word_freqs[word] += 1
        
        vocab = [self.PAD_TOKEN, self.UNK_TOKEN]
        for word in word_freqs:
            for letter in word:
                if letter not in vocab:
                    vocab.append(letter)
                    
        return vocab, word_freqs
    
    def _get_best_pair(self, pair_to_words: Dict[Tuple[str, str], Dict[str, int]], 
                       word_freqs: Dict[str, int]):
        best_pair, best_cntr = None, None
        for pair in pair_to_words:
            cntr = 0
            for word, num in pair_to_words[pair].items():
                cntr += num * word_freqs[word]
                
            if best_pair is None or best_cntr < cntr:
                best_pair = pair
                best_cntr = cntr
        
        return best_pair

    def _update_stats(self, pair_to_words: Dict[Tuple[str, str], Dict[str, int]], 
                      splits: Dict[str, List[str]], best_pair: Tuple[str, str]):
        """
        Updates state of the pair_to_words and splits dictionaries
        (by changing frequency of the affected pairs, adding new pairs
        and modifying splits containing best_pair)
        """
        
        left, right = best_pair
        new_token = left + right
        
        for word in pair_to_words[best_pair].keys():
            split = splits[word]
            
            i = 0
            while i < len(split) - 1:
                if (split[i], split[i + 1]) == best_pair:
                    if 0 < i:
                        pair_to_words[(split[i - 1], new_token)][word] += 1
                        pair_to_words[(split[i - 1], left)][word] -= 1
                        
                        if pair_to_words[(split[i - 1], left)][word] == 0:
                            del pair_to_words[(split[i - 1], left)][word]
                        if len(pair_to_words[(split[i - 1], left)]) == 0:
                            del pair_to_words[(split[i - 1], left)]
                        
                    if i + 2 < len(split):
                        pair_to_words[(new_token, split[i + 2])][word] += 1
                        pair_to_words[(right, split[i + 2])][word] -= 1
                        
                        if pair_to_words[(right, split[i + 2])][word] == 0:
                            del pair_to_words[(right, split[i + 2])][word]
                        if len(pair_to_words[(right, split[i + 2])]) == 0:
                            del pair_to_words[(right, split[i + 2])]
                    
                    split = split[:i] + [new_token] + split[i + 2:]
                    splits[word] = split
                    
                else:
                    i += 1

        del pair_to_words[best_pair]
