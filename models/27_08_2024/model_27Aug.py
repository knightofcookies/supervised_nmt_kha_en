import math
import re
import gc
import unicodedata
import datetime
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast


SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2
UNK_TOKEN = 3


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD", 3: "UNK"}  # Add UNK token
        self.n_words = 4  # Start with 4 to include UNK

    def add_sentence(self, sentence):
        for word in sentence.split(" "):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2count:
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1

    def build_vocab(self, max_vocab_size=25000):
        if max_vocab_size <= self.n_words:
            raise ValueError(
                "max_vocab_size must be greater than the number of existing tokens"
            )

        # Sort words by frequency in descending order
        sorted_words = sorted(self.word2count, key=self.word2count.get, reverse=True)

        # Add most frequent words to vocabulary
        if max_vocab_size < len(sorted_words):
            limit = max_vocab_size - self.n_words
        else:
            limit = len(sorted_words)
        for word in sorted_words[:limit]:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


def unicode_to_ascii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def normalize_string(s):
    if s is not None:
        s = unicode_to_ascii(str(s).lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
        return s.strip()
    else:
        return ""


def read_langs():
    print("Reading lines...")

    with open("../../datasets/pc_30k.tsv", "r", encoding="utf-8") as f:
        lines = f.readlines()

    kha_lines = []
    en_lines = []

    for line in lines[1:]:
        en_line, kha_line = line.split("\t\t\t\t\t")
        en_lines.append(en_line)
        kha_lines.append(kha_line)

    kha_lang = Lang("khasi")
    en_lang = Lang("english")

    return (
        kha_lang,
        en_lang,
        kha_lines,
        en_lines,
    )


def prepare_data():
    (kha_lang, en_lang, kha_lines, en_lines) = read_langs()

    kha_lines = [normalize_string(line) for line in kha_lines]
    en_lines = [normalize_string(line) for line in en_lines]

    print("Read %s sentence pairs" % len(kha_lines))
    print("Counting words...")
    for kha_line in kha_lines:
        kha_lang.add_sentence(normalize_string(kha_line))
    for en_line in en_lines:
        en_lang.add_sentence(normalize_string(en_line))

    kha_lang.build_vocab()
    en_lang.build_vocab()

    print("Counted words:")
    print(kha_lang.name, kha_lang.n_words)
    print(en_lang.name, en_lang.n_words)
    return (kha_lang, en_lang, kha_lines, en_lines)


class ParallelDataset(Dataset):
    def __init__(self, src_lines, tgt_lines, src_lang, tgt_lang):
        self.src_lines = src_lines
        self.tgt_lines = tgt_lines
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, index):
        src_line = self.src_lines[index].strip()
        tgt_line = self.tgt_lines[index].strip()

        src_indices = [
            self.src_lang.word2index.get(word, UNK_TOKEN) for word in src_line.split()
        ] + [EOS_TOKEN]
        tgt_indices = [
            self.tgt_lang.word2index.get(word, UNK_TOKEN) for word in tgt_line.split()
        ] + [EOS_TOKEN]

        return torch.tensor(src_indices), torch.tensor(tgt_indices)


def pad_sequences(batch):
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
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
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
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size).to(DEVICE)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Seq2SeqTransformer(nn.Module):
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
        """
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        return self.transformer.encoder(src_emb, src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
        """
        Decodes the target sequence using the transformer decoder.
        """
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        return self.transformer.decoder(tgt_emb, memory, tgt_mask)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == 0).transpose(0, 1)
    tgt_padding_mask = (tgt == 0).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def train_epoch(
    model,
    optimizer,
    train_dataloader: DataLoader,
    loss_fn,
    scaler: GradScaler,
    reverse=False,
):

    model.train()
    losses = 0

    for src, tgt in train_dataloader:

        if reverse:
            src = tgt.to(DEVICE)
            tgt = src.to(DEVICE)
        else:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

        optimizer.zero_grad()

        with autocast():
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
            tgt_out = tgt[1:, :].long()
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses += loss.item()

    gc.collect()
    torch.cuda.empty_cache()
    return losses / len(list(train_dataloader))


if __name__ == "__main__":

    (
        kha_lang,
        en_lang,
        kha_lines,
        en_lines,
    ) = prepare_data()

    BATCH_SIZE = 32

    parallel_dataset = ParallelDataset(kha_lines, en_lines, kha_lang, en_lang)

    parallel_dataloader = DataLoader(
        parallel_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=pad_sequences,
        drop_last=True,
    )

    HIDDEN_SIZE = 512

    kha_vocab_size = kha_lang.n_words
    en_vocab_size = en_lang.n_words

    kha_to_en_model = Seq2SeqTransformer(
        num_encoder_layers=3,
        num_decoder_layers=3,
        hidden_size=HIDDEN_SIZE,
        nhead=8,
        src_vocab_size=kha_vocab_size,
        tgt_vocab_size=en_vocab_size,
        dim_feedforward=512,
    )

    for p in kha_to_en_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    kha_to_en_model.load_state_dict(torch.load("kha_to_en_model.pth"))
    kha_to_en_model = kha_to_en_model.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    kha_to_en_optimizer = torch.optim.Adam(
        kha_to_en_model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
    )

    pytorch_total_params = sum(p.numel() for p in kha_to_en_model.parameters())
    pytorch_trainable_params = sum(
        p.numel() for p in kha_to_en_model.parameters() if p.requires_grad
    )

    print(f"Total parameters: {pytorch_total_params}")
    print(f"Trainable parameters: {pytorch_trainable_params}")


if __name__ == "__main__":

    NUM_EPOCHS = 1

    scaler = GradScaler()

    for epoch in range(NUM_EPOCHS):

        start = datetime.datetime.now()

        loss_kha_to_en = train_epoch(
            kha_to_en_model,
            kha_to_en_optimizer,
            parallel_dataloader,
            loss_fn,
            scaler,
            False,
        )

        end = datetime.datetime.now()

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss kha -> en : {loss_kha_to_en:.4f}, {end-start}"
        )
        torch.save(kha_to_en_model.state_dict(), "kha_to_en_model.pth")
