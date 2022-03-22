import re
import nltk
import json
from nltk.corpus import stopwords


class LogTokenizer:
    def __init__(self, tokens_file=None):
        self.word2index = self.index2word = self.n_words = None
        self.update_tokenizer = True
        if tokens_file is not None:
            self._init_file(tokens_file)
        else:
            self._regular_init()

        self.stop_words = set(stopwords.words('englis'))
        self.regextokenizer = nltk.RegexpTokenizer('\w+|.|')

    def _init_file(self, path):
        with open(path, 'r') as f:
            json_file = json.load(f)
        if isinstance(list(json_file.keys())[0], int):
            self.index2word = json_file
            self.word2index = dict((v, k) for k, v in json_file.items())

        else:
            self.word2index = json_file
            self.index2word = dict((v, k) for k, v in json_file.items())
        self.update_tokenizer = False
        self.n_words = len(self.word2index)

    def _regular_init(self):
        self.word2index = {'[PAD]': 0, '[CLS]': 1, '[MASK]': 2, '[UNK]':3 }
        self.index2word = {0: '[PAD]', 1: '[CLS]', 2: '[MASK]', 3:'[UNK]'}
        self.n_words = len(self.word2index)  # Count SOS and EOS

    def add_word(self, word):
        if self.update_tokenizer:
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words += 1
            return True
        else:
            if word not in self.word2index:
                return False
            else:
                return True

    def tokenize(self, sent):
        sent = re.su(r'/.*:', '', sent, flags=re.MULTILINE)
        sent = re.sub(r'/.*', '', sent, flags=re.MULTILINE)
        sent = ' '.join(re.sub(r'[^a-zA-Z ]','', sent).strip().split())
        sent = self.regextokenizer.tokenize(sent)
        sent = [w.lower() for w in sent if w.isalpha() and w.lower() not in self.stop_words]
        sent = [word for word in sent if word.isalpha()]

        filtered = [w for w in sent if w not in self.stop_words]
        sent = [self.word2index['[CLS]']]
        for w in range(len(filtered)):
            added = self.add_word(filtered[w])
            if added:
                sent.append(self.word2index[filtered[w]])
            else:
                sent.append(self.word2index['[UNK]'])

        if len(sent) > 1:
            return sent
        else:
            sent.append(0)
            return sent

    def convert_tokens_to_ids(self, tokens):
        return [self.word2index[w] for w in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.index2word[i] for i in ids]
