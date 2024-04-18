import re
import numpy as np
from collections import Counter


def read_corpus(filename: str, padding_token: int = 1) -> list[str]:
    """
    Read and preprocess corpus file
    """
    corpus = []
    for line in open(filename, 'r'):
        # strip white characters
        line = line.strip()
        # substitute punctuation except for ,.?!
        line = re.sub(r'[^\w\s,.?!]|_', '', line)
        # convert lower case & add special tokens to the beginning and end of each line
        line = "<START> " * padding_token + line.lower() + " <END>" * padding_token
        corpus.append(line)
    return corpus


def train_test_split(corpus: list[str], y: list[int], test_ratio: float, shuffle: bool = False):
    """
    Split corpus into train and test set
    """
    labels = np.array(y, dtype=np.int32)
    train_idx, test_idx = np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    for label in set(y):
        idx = np.argwhere(labels == label).squeeze()
        if shuffle:
            np.random.shuffle(idx)
        train_size = int(np.round(len(idx) * (1 - test_ratio)))
        train_idx = np.hstack((train_idx, idx[:train_size]))
        test_idx = np.hstack((test_idx, idx[train_size:]))
    return ([corpus[i] for i in train_idx],  # X_train
            [corpus[i] for i in test_idx],  # X_test
            [y[i] for i in train_idx],  # y_train
            [y[i] for i in test_idx])  # y_test


def vocabulary(train_corpus: list[str], test_corpus: list[str] = None) -> list[str]:
    """
    Create vocabulary from train and test corpus
    """
    vocab = set()
    for paragraph in train_corpus:
        tokens = paragraph.split()
        vocab.update(tokens)
    if test_corpus:
        for paragraph in test_corpus:
            tokens = paragraph.split()
            vocab.update(tokens)
    return list(vocab)


def oov_rate_n_gram(train_corpus: list[str], test_corpus: list[str], n: int) -> float:
    """
    Calculate out-of-vocabulary rate for n-gram model
    """
    train_dict = n_gram_dict(train_corpus, n)
    test_dict = n_gram_dict(test_corpus, n)
    oov_count = 0
    for n_gram in test_dict.keys():
        if n_gram not in train_dict.keys():
            oov_count += test_dict[n_gram]
    return oov_count / sum(test_dict.values())


def n_gram_dict(corpus, n: int) -> dict[str, int]:
    """
    Create n-gram dictionary
    """
    gram_dict = {}
    for paragraph in corpus:
        tokens = paragraph.split()
        for i in range(n, len(tokens) + 1):
            n_gram = tuple(tokens[i - n:i])
            if n_gram in gram_dict.keys():
                gram_dict[n_gram] += 1
            else:
                gram_dict[n_gram] = 1
    return gram_dict


def accuracy(y_pred, y_true):
    """
    Calculate accuracy metric
    """
    return sum(np.array(y_pred) == np.array(y_true)) / len(y_true)


def softmax(x):
    """
    Compute softmax values for each sets of scores in x
    """
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)


class NGramModel:
    """
    N-gram model for classification & text generation
    """

    def __init__(self, corpus: list[str], n: int, y: list[int] = None, V: list[str] = None):
        self.n = n
        self.corpus = np.array(corpus)
        self.y = np.array(y if y else [0 for _ in range(len(corpus))])
        self.category_dict = Counter(self.y)
        self.vocabulary = V if V else vocabulary(corpus)
        self.vocabulary_size = len(self.vocabulary)
        self.conditional_prob_dict = {}

    # classification methods
    def conditional_prob(self):
        """
        Calculate conditional probability of N-gram & (N-1)-gram for each category
        """
        prob_dict = {}
        for label in self.category_dict.keys():
            index = np.argwhere(self.y == label).squeeze()
            prob_dict[label] = {self.n: n_gram_dict(self.corpus[index], self.n),
                                self.n - 1: n_gram_dict(self.corpus[index], self.n - 1)}
        return prob_dict

    def train(self):
        """
        Train the model
        """
        self.conditional_prob_dict = self.conditional_prob()

    def predict(self, test_corpus: list[str]) -> list:
        """
        Predict the label of test corpus
        """
        output = []
        for s in test_corpus:
            words = s.split()
            probs = []
            labels = list(self.category_dict.keys())
            for label in labels:
                logit_P_y = np.log(self.category_dict[label] / len(self.y))
                logit_P_s_y = self.string_prob_logit(words, label)
                probs.append(logit_P_y + logit_P_s_y)
                test = np.sum(logit_P_s_y == 0)
                if test:
                    print(test)
            output.append(labels[np.argmax(probs)])
        return output

    def smoothing_word_prob(self, n_gram: tuple[str], label: int) -> float:
        """
        Calculate the probability of last word in n-gram model with Laplace smoothing
        """
        n_gram_freq = prev_gram_freq = 0
        if n_gram in self.conditional_prob_dict[label][self.n].keys():
            n_gram_freq = self.conditional_prob_dict[label][self.n][n_gram]
        if n_gram[:-1] in self.conditional_prob_dict[label][self.n - 1].keys():
            prev_gram_freq = self.conditional_prob_dict[label][self.n - 1][n_gram[:-1]]
        return (n_gram_freq + 1) / (prev_gram_freq + self.vocabulary_size)

    def string_prob_logit(self, words: list[str], label: int) -> float:
        """
        Calculate the logit probability of a string
        """
        prob_logit = 0
        for i in range(self.n, len(words) + 1):
            n_gram = tuple(words[i - self.n:i])
            prob_logit += np.log(self.smoothing_word_prob(n_gram, label))
        return prob_logit

    # text generation methods
    def generate_sentence(self, count: int, T: float, label: int = None) -> str:
        """
        Generate a sentence
        """
        label = label if label else 0
        words = ["<START>"]
        prev_gram = ["<START>"] * (self.n - 1)
        next_word = ""
        while count and next_word != "<END>":
            next_word = self.generate_word(prev_gram, label, T)
            words.append(next_word)
            prev_gram.pop(0)
            prev_gram.append(next_word)
            count -= 1
        return " ".join(words)

    def generate_word(self, prev_gram: list[str], label: int, T: float) -> str:
        """
        Calculate the probability distribution of the next word to generate one word
        """
        freqs = []
        for v in self.vocabulary:
            n_gram = tuple(prev_gram + [v])
            n_gram_freq = 0
            if n_gram in self.conditional_prob_dict[label][self.n].keys():
                n_gram_freq = self.conditional_prob_dict[label][self.n][n_gram]
            freqs.append(n_gram_freq)
        probs = softmax(np.array(freqs) / T)
        selection_idx = np.random.choice(np.arange(len(probs)), size=1, p=probs).squeeze()
        return self.vocabulary[selection_idx]
