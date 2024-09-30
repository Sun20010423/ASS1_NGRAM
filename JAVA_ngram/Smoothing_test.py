import re
from nltk import ngrams
from nltk.probability import FreqDist
from sklearn.model_selection import train_test_split
from N_gram_show import collect_all_methods
import math
from transformers import BartTokenizer
from tqdm import tqdm


class NGramModel:
    def __init__(self, tokens, n):
        self.n = n
        self.ngram_freq = None
        self.vocab_size = 0
        self.build_model(tokens)

    def build_model(self, tokens):
        ngrams_list = list(ngrams(tokens, self.n))
        self.ngram_freq = FreqDist(ngrams_list)
        self.vocab_size = len(set(tokens))

    def get_ngram_count(self, ngram):
        return self.ngram_freq[ngram]

    def get_context_count(self, context):
        # calculate n-1 gram appeared times in n-1 gram  计算上下文的 n-1 gram 出现的总次数
        return sum(self.ngram_freq[context + (token,)] for token in self.ngram_freq.keys() if token[:-1] == context)

    def calculate_smoothed_probability(self, ngram):
        # laplace smoothed probability  拉普拉斯平滑概率
        context = ngram[:-1]
        ngram_count = self.get_ngram_count(ngram)
        context_count = self.get_context_count(context)

        # laplace smoothed probability  拉普拉斯平滑
        probability = (ngram_count + 1) / (context_count + self.vocab_size)
        return probability

    def calculate_perplexity(self, test_data):
        total_log_prob = 0
        total_ngrams = 0
        print("calculating perplexity...")
        for method in tqdm(test_data):
            tokens = tokenizer(method, add_special_tokens=False)["input_ids"]
            # tokens = re.findall(r'\b\w+\b', method)
            ngrams_list = list(ngrams(tokens, self.n))

            for ngram in ngrams_list:
                smoothed_prob = self.calculate_smoothed_probability(ngram)
                if smoothed_prob > 0:
                    total_log_prob += math.log(smoothed_prob, 2)
                total_ngrams += 1

        if total_ngrams > 0:
            perplexity = 2 ** (-total_log_prob / total_ngrams)
        else:
            perplexity = float('inf')
        return perplexity


def calculate_accuracy(model, test_data, n):
    correct_predictions = 0
    total_predictions = len(test_data)
    print("calculating accuracy...")
    for method in tqdm(test_data):
        tokens = tokenizer(method, add_special_tokens=False)["input_ids"]
        # tokens = re.findall(r'\b\w+\b', method)
        ngrams_list = list(ngrams(tokens, n))

        total_predictions += len(ngrams_list)

        for ngram in ngrams_list:
            prob = model.get_ngram_count(ngram)
            # 这里可以用平滑概率来衡量
            smoothed_prob = model.calculate_smoothed_probability(ngram)
            if smoothed_prob > 0:
                correct_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy


path = "bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(path)
if __name__ == '__main__':

    # Collect all Java Methods 收集所有Java方法
    cleaned_methods = collect_all_methods("actors")
    # Split the dataset into training, validation, and test sets  将数据集划分为训练集、验证集和测试集
    train_data, test_data = train_test_split(cleaned_methods, test_size=0.2, random_state=42)

    # use training set to build n-gram model   使用训练集构建 n-gram 模型
    all_tokens_train = []
    for method in tqdm(train_data):
        tokens = tokenizer(method, add_special_tokens=False)["input_ids"]
        # tokens = re.findall(r'\b\w+\b', method)
        all_tokens_train.extend(tokens)

    # Define the n-gram orders to be compared. 定义要比较的 n-gram 阶数
    n_values = [1, 2, 3]

    # 存储模型结果
    results = []

    for n in n_values:
        ngram_model = NGramModel(all_tokens_train, n)

        # use training set to calculate accuracy 使用测试集计算准确率
        accuracy = calculate_accuracy(ngram_model, test_data, n)
        print(f"n-gram {n} Model accuracy on the test set.: {accuracy:.4f}")  # 模型在测试集上的准确率

        # calculate perplexity 计算困惑度
        perplexity = ngram_model.calculate_perplexity(test_data)
        print(f"n-gram {n} Model perplexity on the test set.: {perplexity:.4f}")

        # save result 保存结果
        results.append({
            'n': n,
            'accuracy': accuracy,
            'perplexity': perplexity
        })

        # Output the most common N-grams built by the model and their smoothed probabilities. 输出模型建立的最常见 N-grams 及其平滑概率
        print(f"present part n-gram {n} model and smoothed probability...")
        for ngram, freq in ngram_model.ngram_freq.most_common(20):
            smoothed_prob = ngram_model.calculate_smoothed_probability(ngram)
            word_list = []
            for x in ngram:
                words = tokenizer.decode(x)
                word_list.append(words)
            print(f"{tuple(word_list)} : {freq}, smoothed probability: {smoothed_prob:.6f}")
        print("\n" + "-" * 80 + "\n")

    # print final results 打印最终结果
    print("Model comparison results :")
    for result in results:
        print(f"n-gram {result['n']} - accuracy: {result['accuracy']:.4f}, perplexity: {result['perplexity']:.4f}")

    # get best model 选择最佳模型
    best_model = min(results, key=lambda x: x['perplexity'])
    print(f"\nbest model: n-gram {best_model['n']} - accuracy: {best_model['accuracy']:.4f}, perplexity: {best_model['perplexity']:.4f}")
