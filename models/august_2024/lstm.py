import unicodedata
import re
import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD"}
        self.n_words = 3

    def add_sentence(self, sentence):
        for word in sentence.split(" "):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


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

    with open("parallel_corpus.tsv", "r", encoding="utf-8") as f:
        lines = f.readlines()

    kha_lines = []
    en_lines = []

    for line in lines:
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

        src_indices = [self.src_lang.word2index[word] for word in src_line.split()] + [
            EOS_TOKEN
        ]
        tgt_indices = [self.tgt_lang.word2index[word] for word in tgt_line.split()] + [
            EOS_TOKEN
        ]

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


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(
            embedding_dim, hidden_size, num_layers, dropout=dropout, bidirectional=True
        )

    def forward(self, input, lengths):
        embedded = self.embedding(input)
        packed_embedded = pack_padded_sequence(embedded, lengths, enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.rnn(packed_embedded)
        outputs, _ = pad_packed_sequence(packed_outputs)

        # Concatenate forward and backward hidden states
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        cell = torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1)

        return outputs, (hidden, cell)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hidden, cell, encoder_outputs):
        embedded = self.embedding(input.unsqueeze(0))
        output, (hidden, cell) = self.rnn(
            embedded, (hidden.unsqueeze(0), cell.unsqueeze(0))
        )
        prediction = self.out(output.squeeze(0))
        return prediction, hidden.squeeze(0), cell.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)

        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.out.out_features

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(src.device)

        input = trg[0, :]  # <SOS> token

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[t] = output
            teacher_force = torch.rand(1) < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs


# Training loop
def train(model, dataloader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for i, (src, tgt) in enumerate(dataloader):
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            src_lengths = torch.tensor(
                [torch.count_nonzero(seq) for seq in src],
                dtype=torch.long,
                device=DEVICE,
            )

            optimizer.zero_grad()
            output = model(src, src_lengths, tgt)

            # Reshape output and target for loss calculation
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            tgt = tgt[1:].view(-1)

            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch: {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f}")


# Example usage
if __name__ == "__main__":
    kha_lang, en_lang, kha_lines, en_lines = prepare_data()

    BATCH_SIZE = 16
    parallel_dataset = ParallelDataset(kha_lines, en_lines, kha_lang, en_lang)
    parallel_dataloader = DataLoader(
        parallel_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_sequences
    )

    INPUT_DIM = kha_lang.n_words
    OUTPUT_DIM = en_lang.n_words
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    model = Seq2Seq(enc, dec).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {pytorch_total_params}")
    print(f"Trainable parameters: {pytorch_trainable_params}")

    # train(model, parallel_dataloader, optimizer, criterion, epochs=10)  # Example epochs
