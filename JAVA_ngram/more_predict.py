from only_predict import NGramModel, suggest_completion
from N_gram_show import collect_all_methods
from transformers import BartTokenizer

path = "bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(path)

processed_data_list = collect_all_methods("actors")
n = 2  # bigram
ngram_model = NGramModel(n)
test_100 = []
for processed_data in processed_data_list:
    test_100.append(processed_data.split()[0])
    ngram_model.train(processed_data)

for index, prefix in enumerate(test_100[17500:17600]):
    print(f"number {index + 1} input :", prefix)
    suggestions = suggest_completion(ngram_model, prefix)
    print("Suggestions:")
    for suggestion, probability in suggestions[:3]:
        suggestion_decode = tokenizer.decode(suggestion)
        print(f"{suggestion_decode} : {probability:.4f}")
    print("=====================================================================================")
