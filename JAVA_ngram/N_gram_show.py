import os
import re
from nltk import ngrams
from nltk.probability import FreqDist
from transformers import BartTokenizer
from tqdm import tqdm


def remove_comments(code_snippet):
    code_snippet = re.sub(r'import\s+[\w.]+\s*;', '', code_snippet)
    code_snippet = re.sub(r'package\s+[\w.]+\s*;', '', code_snippet)
    code_without_single_line_comments = re.sub(r'//.*?(\n|$)', r'\1', code_snippet, flags=re.DOTALL)
    code_without_comments = re.sub(r'/\*.*?\*/', '', code_without_single_line_comments, flags=re.DOTALL)
    code_without_comments = re.sub(r'/\*.*$', '', code_without_comments, flags=re.DOTALL)
    code_without_star_lines = re.sub(r'^\s*\*.*?(\n|$)', r'\1', code_without_comments, flags=re.MULTILINE | re.DOTALL)
    code_without_single_line_comments = re.sub(r'//.*?(\n|$)', r'\1', code_without_star_lines)
    code_without_single_line_comments = re.sub(r'//.*', '', code_without_single_line_comments)
    code_without_comments = re.sub(r'/\*.*?\*/', '', code_without_single_line_comments, flags=re.DOTALL)
    line = " ".join(code_without_comments.split())
    line = line.replace("}", "")
    return line


def extract_java_methods(file_path):
    content_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.readlines()
        for line_list in content:
            line_list = remove_comments(line_list)
            if line_list:
                content_list.append(line_list)
    return content_list


def collect_all_methods(root_dir):
    all_methods = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".java"):
                file_path = os.path.join(subdir, file)
                methods = extract_java_methods(file_path)
                all_methods.extend(methods)
    print(f"extract {len(all_methods)} methods")
    return all_methods


class NGramModel:
    def __init__(self, tokens, n):
        self.n = n
        self.ngram_freq = None
        self.vocab_size = 0
        self.build_model(tokens)

    def build_model(self, tokens):
        # Build n-grams   生成 n-grams
        ngrams_list = list(ngrams(tokens, self.n))
        self.ngram_freq = FreqDist(ngrams_list)
        self.vocab_size = len(set(tokens))  # Vocabulary Size 词汇表大小

    def get_ngram_count(self, ngram):
        return self.ngram_freq[ngram]

    def get_context_count(self, ngram):
        if self.n == 1:
            return sum(self.ngram_freq.values())
        context_ngram = ngram[:-1]  # Get Context  获取上下文
        return sum(count for ngram_key, count in self.ngram_freq.items() if ngram_key[:-1] == context_ngram)

    def calculate_smoothed_probability(self, ngram):
        ngram_count = self.get_ngram_count(ngram)
        context_count = self.get_context_count(ngram)

        # Laplace smoothing probability 拉普拉斯平滑概率
        probability = (ngram_count + 1) / (context_count + self.vocab_size)
        return probability


def calculate_accuracy(model, test_data, n):
    correct_predictions = 0
    total_predictions = len(test_data)

    for method in tqdm(test_data):
        tokens = re.findall(r'\b\w+\b', method)
        ngrams_list = list(ngrams(tokens, n))

        for ngram in ngrams_list:
            prob = model.get_ngram_count(ngram)
            if prob > 0:
                correct_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy


path = "bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(path)
if __name__ == '__main__':

    cleaned_methods = collect_all_methods("actors")
    all_tokens = []
    for method in tqdm(cleaned_methods):
        tokens = tokenizer(method, add_special_tokens=False)["input_ids"]
        # tokens = re.findall(r'\b\w+\b', method)
        all_tokens.extend(tokens)

    n = 2
    ngram_model = NGramModel(all_tokens, n)
    # print(f"Top 10 n-grams: {ngram_model.ngram_freq.most_common(10)}")
    print("Top 10 n-grams:")
    for ngram, freq in ngram_model.ngram_freq.most_common(10):
        word_list = [tokenizer.decode(i).strip() for i in ngram]
        print(f"{tuple(word_list)} : {freq}")

    # calculate and present every n-gram's smoothed probability 计算并展示每个 n-gram 的平滑概率
    #print("展示部分N-gram模型的平滑概率...")
    print("present part of N-gram smoothed probability...")
    for ngram, freq in ngram_model.ngram_freq.most_common(20):
        prob = ngram_model.calculate_smoothed_probability(ngram)
        word_list = []
        for i in ngram:
            word = tokenizer.decode(i)
            word_list.append(word)
        print(f"{tuple(word_list)} : {freq}, smoothed probability: {prob:.6f}")
