"""
This module implements a sequence-to-sequence transformer model for machine translation. 
It uses a parallel corpus to train the model and provides functionality for translating sentences between two languages.

The module includes the following components:

- **Lang Class:** Manages vocabulary and word-to-index mappings for a language.
- **Data Preprocessing Functions:** Functions for normalizing strings and preparing the data for training.
- **ParallelDataset Class:** A PyTorch Dataset class for parallel corpus data.
- **Padding Function:** A function to pad sequences to the same length.
- **PositionalEncoding Class:** Implements positional encoding as described in the "Attention is All You Need" paper.
- **TokenEmbedding Class:** Embeds tokens into a vector space.
- **Seq2SeqTransformer Class:** The main sequence-to-sequence transformer model.
- **Masking Functions:** Functions for creating masks for the source and target sequences.
- **Training Function:** A function to train the model for one epoch.
- **Decoding Function:** A function to perform greedy decoding to generate a target sequence.
- **Translation Function:** A function to translate a source sentence into a target sentence.

The main script trains a model to translate from Khasi to English. It uses a parallel corpus stored in a TSV file. 
The trained model can then be used to translate new sentences from Khasi to English.
"""

import math
import re
import unicodedata
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Lang:
    """
    Helper class to manage vocabulary and word-to-index mappings for a language.
    """

    def __init__(self, name):
        """
        Initializes a new Lang object.

        Args:
            name: The name of the language.
        """
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD"}
        self.n_words = 3

    def add_sentence(self, sentence):
        """
        Adds a sentence to the vocabulary.

        Args:
            sentence: The sentence to add.
        """
        for word in sentence.split(" "):
            self.add_word(word)

    def add_word(self, word):
        """
        Adds a word to the vocabulary.

        Args:
            word: The word to add.
        """
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicode_to_ascii(s):
    """
    Converts a unicode string to ASCII.

    Args:
        s: The unicode string to convert.

    Returns:
        The ASCII representation of the string.
    """
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def normalize_string(s):
    """
    Normalizes a string by converting it to lowercase, removing punctuation, and trimming whitespace.

    Args:
        s: The string to normalize.

    Returns:
        The normalized string.
    """
    if s is not None:
        s = unicode_to_ascii(str(s).lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
        return s.strip()
    else:
        return ""


def read_langs():
    """
    Reads the parallel corpus and creates Lang objects for the source and target languages.

    Returns:
        A tuple containing the source language object, target language object, source lines, and target lines.
    """
    print("Reading lines...")

    df = pd.read_csv("test_pc.tsv", sep="\t\t\t\t\t")

    kha_lines = df["kha"].values
    en_lines = df["en"].values

    kha_lang = Lang("khasi")
    en_lang = Lang("english")

    return (
        kha_lang,
        en_lang,
        kha_lines,
        en_lines,
    )


def prepare_data():
    """
    Prepares the data for training by reading the parallel corpus, normalizing the strings, and creating Lang objects.

    Returns:
        A tuple containing the source language object, target language object, normalized source lines, and normalized target lines.
    """
    (kha_lang, en_lang, kha_lines, en_lines) = read_langs()

    kha_lines = [normalize_string(line) for line in kha_lines]
    en_lines = [normalize_string(line) for line in en_lines]

    print("Read %s sentence pairs" % len(kha_lines))
    print("Counting words...")
    for kha_line in kha_lines:
        kha_lang.add_sentence(normalize_string(kha_line))
    for en_line in en_lines:
        en_lang.add_sentence(normalize_string(en_line))
    print("Counted words:")
    print(kha_lang.name, kha_lang.n_words)
    print(en_lang.name, en_lang.n_words)
    return (kha_lang, en_lang, kha_lines, en_lines)


class ParallelDataset(Dataset):
    """
    Dataset class for parallel corpus data.
    """

    def __init__(self, src_lines, tgt_lines, src_lang, tgt_lang):
        """
        Initializes a new ParallelDataset object.

        Args:
            src_lines: List of source language sentences.
            tgt_lines: List of target language sentences.
            src_lang: Source language object (Lang).
            tgt_lang: Target language object (Lang).
        """
        self.src_lines = src_lines
        self.tgt_lines = tgt_lines
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.src_lines)

    def __getitem__(self, index):
        """
        Returns a pair of source and target tensors for the given index.

        Args:
            index: The index of the data point to retrieve.

        Returns:
            A tuple containing the source tensor and target tensor.
        """
        src_line = self.src_lines[index].strip()
        tgt_line = self.tgt_lines[index].strip()

        src_indices = [self.src_lang.word2index[word] for word in src_line.split()] + [
            EOS_TOKEN
        ]
        tgt_indices = [self.tgt_lang.word2index[word] for word in tgt_line.split()] + [
            EOS_TOKEN
        ]

        return torch.tensor(src_indices), torch.tensor(tgt_indices)


def pad_sequences(batch):
    """
    Pads a batch of sequences to the same length.

    Args:
        batch: A list of tuples, where each tuple contains a source tensor and a target tensor.

    Returns:
        A tuple containing the padded source tensors and padded target tensors.
    """
    # Sort the sequences by the length of the source sequence (descending)
    sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

    # Separate source and target sequences
    src_seqs = [x[0] for x in sorted_batch]
    tgt_seqs = [x[1] for x in sorted_batch]

    # Find the maximum length among both source and target sequences
    max_length = max(max(len(s) for s in src_seqs), max(len(t) for t in tgt_seqs))

    # Pad the sequences to the maximum length
    src_padded = [
        torch.cat([s, torch.tensor([PAD_TOKEN] * (max_length - len(s)))])
        for s in src_seqs
    ]
    tgt_padded = [
        torch.cat([t, torch.tensor([PAD_TOKEN] * (max_length - len(t)))])
        for t in tgt_seqs
    ]

    # Convert to tensors
    src_padded = torch.stack(src_padded)
    tgt_padded = torch.stack(tgt_padded)

    return src_padded, tgt_padded


class PositionalEncoding(nn.Module):
    """
    Implements positional encoding as described in the "Attention is All You Need" paper.
    """

    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        """
        Initializes a new PositionalEncoding object.

        Args:
            emb_size: The embedding size.
            dropout: The dropout probability.
            maxlen: The maximum sequence length.
        """
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        """
        Applies positional encoding to the token embeddings.

        Args:
            token_embedding: The token embeddings.

        Returns:
            The token embeddings with positional encoding applied.
        """
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


class TokenEmbedding(nn.Module):
    """
    Embeds tokens into a vector space.
    """

    def __init__(self, vocab_size: int, emb_size):
        """
        Initializes a new TokenEmbedding object.

        Args:
            vocab_size: The size of the vocabulary.
            emb_size: The embedding size.
        """
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size).to(DEVICE)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        """
        Embeds the tokens into a vector space.

        Args:
            tokens: The tokens to embed.

        Returns:
            The embedded tokens.
        """
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Seq2SeqTransformer(nn.Module):
    """
    Sequence-to-sequence transformer model.
    """

    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        hidden_size: int,
        nhead: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        """
        Initializes a new Seq2SeqTransformer object.

        Args:
            num_encoder_layers: The number of encoder layers.
            num_decoder_layers: The number of decoder layers.
            hidden_size: The hidden size.
            nhead: The number of attention heads.
            src_vocab_size: The size of the source vocabulary.
            tgt_vocab_size: The size of the target vocabulary.
            dim_feedforward: The dimension of the feedforward network.
            dropout: The dropout probability.
        """
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.generator = nn.Linear(hidden_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, hidden_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size, dropout=dropout).to(
            DEVICE
        )

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_padding_mask: torch.Tensor,
        tgt_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ):
        """
        Forward pass of the transformer model.

        Args:
            src: The source sequence.
            trg: The target sequence.
            src_mask: The source mask.
            tgt_mask: The target mask.
            src_padding_mask: The source padding mask.
            tgt_padding_mask: The target padding mask.
            memory_key_padding_mask: The memory key padding mask.

        Returns:
            The output of the transformer model.
        """
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))

        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )
        return self.generator(outs)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        """
        Encodes the source sequence using the transformer encoder.

        Args:
            src: The source sequence.
            src_mask: The source mask.

        Returns:
            The encoded source sequence.
        """
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        return self.transformer.encoder(src_emb, src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
        """
        Decodes the target sequence using the transformer decoder.

        Args:
            tgt: The target sequence.
            memory: The encoded source sequence.
            tgt_mask: The target mask.

        Returns:
            The decoded target sequence.
        """
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        return self.transformer.decoder(tgt_emb, memory, tgt_mask)


def generate_square_subsequent_mask(sz):
    """
    Generates a square subsequent mask of size sz.

    Args:
        sz: The size of the mask.

    Returns:
        The square subsequent mask.
    """
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt):
    """
    Creates masks for the source and target sequences.

    Args:
        src: The source sequence.
        tgt: The target sequence.

    Returns:
        A tuple containing the source mask, target mask, source padding mask, and target padding mask.
    """
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == 0).transpose(0, 1)
    tgt_padding_mask = (tgt == 0).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def train_epoch(model, optimizer, train_dataloader, loss_fn, reverse=False):
    """
    Trains the model for one epoch.

    Args:
        model: The model to train.
        optimizer: The optimizer to use.
        train_dataloader: The data loader for the training data.
        loss_fn: The loss function to use.
        reverse: Whether to reverse the source and target sequences.

    Returns:
        The average loss for the epoch.
    """
    model.train()
    losses = 0

    for src, tgt in train_dataloader:
        if reverse:
            src = tgt.to(DEVICE)
            tgt = src.to(DEVICE)
        else:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input
        )

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )

        optimizer.zero_grad()

        tgt_out = tgt[1:, :].long()

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        loss.backward()

        optimizer.step()

        losses += loss.item()

    return losses / len(list(train_dataloader))


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    Performs greedy decoding to generate a target sequence.

    Args:
        model: The model to use for decoding.
        src: The source sequence.
        src_mask: The source mask.
        max_len: The maximum length of the target sequence.
        start_symbol: The start symbol for the target sequence.

    Returns:
        The generated target sequence.
    """
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(
            DEVICE
        )
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_TOKEN:
            break
    return ys


def translate(
    model: torch.nn.Module, src_sentence: str, src_lang: Lang, tgt_lang: Lang
):
    """
    Translates a source sentence into a target sentence using the provided model.

    Args:
        model: The trained Seq2SeqTransformer model.
        src_sentence: The source sentence to translate.
        src_lang: The source language object (Lang).
        tgt_lang: The target language object (Lang).

    Returns:
        The translated target sentence.
    """
    model.eval()
    src_sentence = normalize_string(src_sentence)
    src_indices = [src_lang.word2index[word] for word in src_sentence.split()] + [
        EOS_TOKEN
    ]
    src = torch.tensor(src_indices).view(-1, 1).to(DEVICE)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(DEVICE)
    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=num_tokens + 5, start_symbol=SOS_TOKEN
    ).flatten()
    return (
        " ".join(tgt_lang.index2word[idx] for idx in list(tgt_tokens.cpu().numpy()))
        .replace("SOS", "")
        .replace("EOS", "")
        .strip()
    )


if __name__ == "__main__":

    (
        kha_lang,
        en_lang,
        kha_lines,
        en_lines,
    ) = prepare_data()

    BATCH_SIZE = 4

    parallel_dataset = ParallelDataset(kha_lines, en_lines, kha_lang, en_lang)

    parallel_dataloader = DataLoader(
        parallel_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=pad_sequences,
        drop_last=True,
    )

    HIDDEN_SIZE = 128

    kha_vocab_size = kha_lang.n_words
    en_vocab_size = en_lang.n_words

    kha_to_en_model = Seq2SeqTransformer(
        num_encoder_layers=3,
        num_decoder_layers=3,
        hidden_size=HIDDEN_SIZE,
        nhead=8,
        src_vocab_size=kha_vocab_size,
        tgt_vocab_size=en_vocab_size,
    )

    kha_to_en_model = kha_to_en_model.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    kha_to_en_optimizer = torch.optim.Adam(
        kha_to_en_model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
    )

    NUM_EPOCHS = 100

    best_loss = 1e6

    for epoch in range(NUM_EPOCHS):

        loss_kha_to_en = train_epoch(
            kha_to_en_model,
            kha_to_en_optimizer,
            parallel_dataloader,
            loss_fn,
            False,
        )

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss kha -> en : {loss_kha_to_en:.4f}"
            )
            if loss_kha_to_en < best_loss:
                best_loss = loss_kha_to_en
                torch.save(kha_to_en_model.state_dict(), "kha_to_en_model.pth")


    print(f"Best model has loss: {best_loss:.4f}")

    print(
        translate(
            kha_to_en_model, "Ki jingkhynñiuh jumai ha New Zealand", kha_lang, en_lang
        )
    )
