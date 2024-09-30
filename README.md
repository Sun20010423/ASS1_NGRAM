# N-Gram Model Training and Evaluation

This project implements an N-Gram language model to predict the next token in a sequence of Java code snippets. The process involves data cleaning, tokenization using Hugging Face’s BART tokenizer, and training the model using N-grams with various values of n. The project not only focuses on model training but also includes model evaluation using metrics such as accuracy and perplexity. Additionally, multiple N-gram models (1-gram, 2-gram, 3-gram) are compared to identify the best-performing model. The results are validated through accuracy and perplexity calculations, providing a comprehensive analysis of the model's predictive capabilities.

## Prerequisites
Before running the script, ensure you have the following installed:

- Python 3.7+
- Required libraries, which can be installed by running:

The following Python libraries are required:
  - transformers
  - tqdm
  - nltk
  - scikit-learn
## Project Structure
- `N_gram_show.py`: The main script that extracts Java code snippets, preprocesses them, and builds an N-gram model. It also displays the most common N-grams and their smoothed probabilities.
- `only_predict.py`: This script is focused on generating predictions from an N-gram model given an input prefix.
- `more_predict.py`: This script tests predictions on a subset of Java methods and prints the top suggestions.
- `smoothing_test.py`: This script implements Laplace smoothing for the N-gram model, calculating accuracy and perplexity on a test dataset.
## Instructions
### 1. Preprocessing
First, the code snippets need to be preprocessed by removing comments and other non-essential characters using regular expressions. This is done using the `preprocess_code()` function in `N_gram_show.py`.
### 2. N-Gram Model Training
You can train an N-Gram model using the `train()` function from `NGramModel` in either `N_gram_show.py` or `only_predict.py`. The training process involves tokenizing Java code snippets and constructing N-Grams.
Example:
Train the N-Gram model
```python
ngram_model = NGramModel(n=2)  `# Use bigram model`
for processed_data in processed_data_list:
    ngram_model.train(processed_data)
```
### 3. Prediction
The trained N-Gram model can predict the next token in a sequence given an input prefix using the `predict()` function in `only_predict.py`. The prediction output includes tokens with their respective probabilities.

### 4. Evaluation
The model's performance is evaluated using two metrics:

  - Accuracy: The proportion of correct token predictions over the total.
  - Perplexity: A measure of how well the model predicts a sample, where lower perplexity indicates a better model.
You can compute these using the `calculate_accuracy()` and c`alculate_perplexity()` functions in `smoothing_test.py`.

### 5. Running the Scripts
Run `N_gram_show.py` to preprocess the data and train the N-gram model.
Use `only_predict.py` to generate predictions for a given prefix.
Run `more_predict.py` to test predictions on multiple prefixes.
Run `smoothing_test.py` to evaluate the model on a test dataset using accuracy and perplexity.

## Results
The results of the N-Gram model, including common N-grams, their probabilities, and evaluation metrics, will be printed to the console. Example results include the accuracy and perplexity of the model on a test dataset.


