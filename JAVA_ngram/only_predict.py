import re
from collections import defaultdict
import os
from N_gram_show import collect_all_methods
from transformers import BartTokenizer



def preprocess_code(code_snippets):

    return re.sub(r'[^a-zA-Z0-9_(){}[\];:,.]', ' ', code_snippets)   # remove illegal characters


class NGramModel:
    def __init__(self, n):
        self.n = n
        self.ngrams = defaultdict(int)
        self.total_ngrams = 0

    def train(self, data):
        # tokens = data.split()
        tokens = tokenizer(data, add_special_tokens=False)["input_ids"]
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            self.ngrams[ngram] += 1
            self.total_ngrams += 1

    def calculate_probability(self, ngram):
        if ngram in self.ngrams:
            return (self.ngrams[ngram] + 1) / (self.total_ngrams + len(self.ngrams))
        else:
            return 1 / (self.total_ngrams + len(self.ngrams))

    def predict(self, prefix):
        # prefix_tokens = prefix.split()
        prefix_tokens = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        if len(prefix_tokens) < self.n - 1:
            return []

        last_ngram = tuple(prefix_tokens[-(self.n - 1):])
        candidates = {ngram[-1]: count for ngram, count in self.ngrams.items() if ngram[:-1] == last_ngram}
        predictions = {token: self.calculate_probability(last_ngram + (token,)) for token in candidates}
        return sorted(predictions.items(), key=lambda x: x[1], reverse=True)


def suggest_completion(model, prefix):
    predictions = model.predict(prefix)
    return predictions


path = "bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(path)
if __name__ == '__main__':
    # text preprocessing 预处理文本
    processed_data_list = collect_all_methods("actors")
    # build n-gram model 构建 n-gram 模型
    n = 2  # use bigram or trigram
    ngram_model = NGramModel(n)
    for processed_data in processed_data_list:
        ngram_model.train(processed_data)

    # To Predict  进行预测
    prefix = "public"      # can input prefix  if bigram(n =2) only input 1 word  if trigram(n=3) input 2 words
    print()
    print("input:", prefix)
    print()
    suggestions = suggest_completion(ngram_model, prefix)
    print("Suggestions:")
    for suggestion, probability in suggestions[:5]:
        suggestion_decode = tokenizer.decode(suggestion)
        print(f"{suggestion_decode}:{probability:.4f}")
