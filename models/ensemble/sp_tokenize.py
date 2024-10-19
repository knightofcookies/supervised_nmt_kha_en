import sentencepiece as spm


def train_tokenizer(input_file, model_prefix, vocab_size):
    """Trains a SentencePiece tokenizer and saves the model.

    Args:
        input_files: List of paths to input text files.
        model_prefix: Prefix for the output model files.
        vocab_size: Vocabulary size for the tokenizer.
    """

    spm.SentencePieceTrainer.Train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="unigram",  # or 'unigram', 'char', 'word'
        character_coverage=1.0,  # For languages with large character sets
        #  Add other training parameters as needed (e.g., user_defined_symbols)
    )


def tokenize_file(input_file, model_prefix, output_file):
    """Tokenizes a text file using a trained SentencePiece model.

    Args:
        input_file: Path to the input text file.
        model_prefix: Prefix of the trained SentencePiece model.
        output_file: Path to save the tokenized output.
    """

    sp = spm.SentencePieceProcessor()
    sp.Load(f"{model_prefix}.model")

    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            tokenized_line = sp.EncodeAsPieces(line.strip())
            outfile.write(" ".join(tokenized_line) + "\n")


en_train = "C:\\Users\\ahlad\\Computer Programming\\GitHub\\supervised_nmt_kha_en\\models\\ensemble\\multi30k_train_en.txt"
# en_test = "C:\\Users\\ahlad\\Computer Programming\\GitHub\\supervised_nmt_kha_en\\datasets\\en_nits_test.txt"
de_train = "C:\\Users\\ahlad\\Computer Programming\\GitHub\\supervised_nmt_kha_en\\models\\ensemble\\multi30k_train_de.txt"
# kha_test = "C:\\Users\\ahlad\\Computer Programming\\GitHub\\supervised_nmt_kha_en\\datasets\\kha_nits_test.txt"

en_model_prefix = "en_multi30k_unigram"
kha_model_prefix = "de_multi30k_unigram"
en_vocab_size = 6312
de_vocab_size = 11068

train_tokenizer(en_train, en_model_prefix, en_vocab_size)
train_tokenizer(de_train, kha_model_prefix, de_vocab_size)


tokenize_file(en_train, en_model_prefix, "en_train.tok")
# tokenize_file(en_test, en_model_prefix, "en_test.tok")
tokenize_file(de_train, kha_model_prefix, "de_train.tok")
# tokenize_file(kha_test, kha_model_prefix, "kha_test.tok")


print(
    f"Tokenizer trained and saved to {en_model_prefix}.model and {kha_model_prefix}.model"
)
print("Tokenized files saved with .tok extension.")
