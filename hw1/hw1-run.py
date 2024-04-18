from hw1 import *
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Customize the parameters for n-gram model.")

# Add arguments
parser.add_argument("--human", help="Path to human corpus file", type=str)
parser.add_argument("--gpt", help="Path to gpt corpus file", type=str)
parser.add_argument("-N", help="n-gram model parameter", type=int, default=2)
parser.add_argument("-T", "--temperature", help="temperature parameter", type=float, default=50)
parser.add_argument("-l", "--length", help="sentence length", type=int, default=20)
parser.add_argument("-c", "--count", help="count of word in a sentence", type=int, default=5)
parser.add_argument("--only", help="only generate sentences", type=int, default=0)
# Parse arguments
args = parser.parse_args()

# Parameters parsing
human_file = args.human  # human corpus file path
gpt_file = args.gpt  # gpt corpus file path
N = args.N  # n-gram model parameter
T = args.temperature  # temperature
sentence_len = args.length  # sentence length
n_sentence = args.count  # number of sentences
only_generation = args.only  # only generate sentences


def generate_model(corpus, n):
    """
    Simple wrapper to generate n-gram model
    """
    model = NGramModel(corpus, n)
    model.train()
    return model


print(f"##### {N}-gram model #####")

# 1. Read corpus
human_corpus = read_corpus(human_file, padding_token=N - 1)
gpt_corpus = read_corpus(gpt_file, padding_token=N - 1)
labels = [0 for _ in range(len(human_corpus))]
labels.extend([1 for _ in range(len(gpt_corpus))])
corpus = human_corpus.copy()
corpus.extend(gpt_corpus)


if only_generation:
    print("Skip classification procedure.\n")
else:
    # 2. Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(corpus, labels, test_ratio=0.1)

    # 3. Generate vocabulary
    V = vocabulary(X_train, X_test)

    # 4. Calculate out-of-vocabulary rate
    oov_rate = oov_rate_n_gram(X_train, X_test, n=N)
    print(f"OOV rate for {N}-gram model: {oov_rate * 100: .2f}%")

    # 5. Train 2-gram model
    bigram_model = NGramModel(X_train, n=N, y=y_train, V=V)
    bigram_model.train()

    # 6. Predict and calculate accuracy
    pred = bigram_model.predict(X_test)
    acc = accuracy(pred, y_test)
    print(f"Classification accuracy for {N}-gram model: {acc * 100: .2f}%")


# 7. Generate sentences using 2-gram model
print(f"\nGenerated {sentence_len}-words sentences for {n_sentence} times under {T} temperature with {N}-gram model...")
# human corpus for text generation
human_bigram = generate_model(human_corpus, N)
human_generation = [human_bigram.generate_sentence(count=sentence_len, T=T) for _ in range(n_sentence)]
result1 = "\n".join(human_generation)
print(f"\nHuman corpus: \n{result1}")
# gpt corpus for text generation
gpt_bigram = generate_model(gpt_corpus, N)
gpt_generation = [gpt_bigram.generate_sentence(count=sentence_len, T=T) for _ in range(n_sentence)]
result2 = "\n".join(gpt_generation)
print(f"\nGPT corpus: \n{result2}")

