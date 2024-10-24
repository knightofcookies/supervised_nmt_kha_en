import time
import random
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import sentencepiece as spm
import lib_seq2seq


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


en_model_prefix = "en_multi30k_word"
de_model_prefix = "de_multi30k_word"

en_sp = spm.SentencePieceProcessor()
de_sp = spm.SentencePieceProcessor()

en_sp.Load(f"{en_model_prefix}.model")
de_sp.Load(f"{de_model_prefix}.model")


word_vec_size = 512
en_vocab_size = en_sp.GetPieceSize()
de_vocab_size = de_sp.GetPieceSize()
en_word_padding_idx = en_sp.pad_id()
de_word_padding_idx = de_sp.pad_id()
dropout = 0.1
position_encoding = True


def numericalize(text, tokenizer):
    ids = tokenizer.EncodeAsIds(text)
    return torch.tensor(ids)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.en_embeddings = lib_seq2seq.Embeddings(
            word_vec_size,
            en_vocab_size,
            en_word_padding_idx,
            position_encoding=position_encoding,
            dropout=dropout,
        )
        self.de_embeddings = lib_seq2seq.Embeddings(
            word_vec_size,
            de_vocab_size,
            de_word_padding_idx,
            position_encoding=position_encoding,
            dropout=dropout,
        )
        self.en_encoder = lib_seq2seq.CNNEncoder(
            cnn_kernel_width=3,
            num_layers=20,
            hidden_size=512,
            dropout=0.1,
            embeddings=self.en_embeddings,
        )
        self.de_decoder = lib_seq2seq.CNNDecoder(
            num_layers=20,
            hidden_size=512,
            attn_type="general",
            copy_attn=False,
            cnn_kernel_width=3,
            dropout=0.1,
            embeddings=self.de_embeddings,
            copy_attn_type="general",
        )
        self.output_layer_de = nn.Linear(512, de_vocab_size)
        for embedding in self.en_embeddings.emb_luts:
            init.xavier_uniform_(embedding.weight)
        for embedding in self.de_embeddings.emb_luts:
            init.xavier_uniform_(embedding.weight)

model = Model()

criterion = nn.CrossEntropyLoss(ignore_index=de_word_padding_idx)


def train_en_to_de(
    model: Model,
    criterion,
    optimizer,
    en_sp,
    de_sp,
    num_epochs=10,
    batch_size=32,
):
    def load_data(en_filepath, de_filepath):
        en_data = []
        de_data = []
        with open(en_filepath, "r", encoding="utf-8") as en_f, open(
            de_filepath, "r", encoding="utf-8"
        ) as de_f:
            for en_line, de_line in zip(en_f, de_f):
                en_data.append(en_line.strip())
                de_data.append(de_line.strip())
        return en_data, de_data

    en_train, de_train = load_data("multi30k_train_en.txt", "multi30k_train_de.txt")

    for epoch in range(num_epochs):
        start_time = time.time()
        total_loss = 0

        data = list(zip(en_train, de_train))
        random.shuffle(data)
        en_train, de_train = zip(*data)

        # losses = [] # DEBUG

        for i in range(0, len(en_train), batch_size):
            en_batch = en_train[i : i + batch_size]
            de_batch = de_train[i : i + batch_size]

            # Numericalize and pad the *entire batch*
            en_numericalized = [numericalize(text, en_sp) for text in en_batch]
            de_numericalized = [numericalize(text, de_sp) for text in de_batch]

            en_lengths = torch.tensor([tensor.shape[0] for tensor in en_numericalized])

            en_input = nn.utils.rnn.pad_sequence(
                en_numericalized, padding_value=en_word_padding_idx, batch_first=True
            )
            de_input = nn.utils.rnn.pad_sequence(
                de_numericalized, padding_value=de_word_padding_idx, batch_first=True
            )

            en_input = en_input.to(device)
            de_input = de_input.to(device)
            en_lengths = en_lengths.to(device)

            en_encoded, en_remap, _ = model.en_encoder(en_input, en_lengths)

            # Decoder training (English to German) using Teacher Forcing
            model.de_decoder.init_state(
                None, en_encoded, en_remap
            )  # Use English encoding to initialize German decoder

            de_target_input = de_input[:, :-1].to(
                device
            )  # Shift target for teacher forcing
            de_target_output = de_input[:, 1:].to(device)

            de_decoded_output, _ = model.de_decoder(
                de_target_input, en_encoded, memory_lengths=en_lengths
            )  # Use English encoded output as memory bank
            output = model.output_layer_de(de_decoded_output)

            loss = criterion(
                output.contiguous().view(-1, de_vocab_size),
                de_target_output.contiguous().view(-1),
            )

            optimizer.zero_grad()
            # losses.append((loss.item())) # DEBUG
            loss.backward()
            optimizer.step()
            total_loss += loss.item() / len(en_train)

        end_time = time.time()
        epoch_time = end_time - start_time

        print(
            f"Epoch: {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Time: {epoch_time:.2f}s"
        )
        # print(losses)
        # exit(0)


optimizer_en_de = optim.Adam(
    list(model.parameters()),
    lr=0.0001,
)

model.to(device)
criterion.to(device)

train_en_to_de(
    model,
    criterion,
    optimizer_en_de,
    en_sp,
    de_sp,
    num_epochs=20,
    batch_size=64,
)


def evaluate_en_to_de(
    model: Model,
    en_sp,
    de_sp,
    en_input,
):
    """Evaluate English to German translation."""
    model.eval()
    with torch.no_grad():
        en_numericalized = numericalize(en_input, en_sp).unsqueeze(0).to(device)
        en_length = torch.tensor([en_numericalized.shape[1]]).to(device)
        en_input = nn.utils.rnn.pad_sequence(
            en_numericalized, padding_value=en_word_padding_idx, batch_first=True
        ).to(device)

        en_encoded, en_remap, en_lengths_output = model.en_encoder(en_input, en_length)

        model.de_decoder.init_state(None, en_encoded, en_remap)

        de_decoded_words = []
        de_prev_word = torch.tensor([[de_sp.bos_id()]]).to(device)

        for _ in range(en_length + 5):  # Max output length
            de_decoder_output, _ = model.de_decoder(
                de_prev_word, en_encoded, memory_lengths=en_lengths_output
            )

            output = model.output_layer_de(de_decoder_output)
            de_predicted_word = output.argmax(2).squeeze()

            if de_predicted_word.item() == de_sp.eos_id():
                break

            # de_decoded_words.append(
            #     list(de_vocab.keys())[
            #         list(de_vocab.values()).index(de_predicted_word.item())
            #     ]
            # )
            de_decoded_words.append(de_sp.IdToPiece(de_predicted_word.item()))

            de_prev_word = de_predicted_word.view(1, 1)

    model.train()
    return "".join(de_decoded_words).replace("▁", " ")


en_input_sentence = "A little girl climbing into a wooden playhouse."
translated_sentence_de = evaluate_en_to_de(
    model,
    en_sp,
    de_sp,
    en_input_sentence,
)
print(f"Translated Sentence (German): {translated_sentence_de}")

torch.save(model, "model_20.pt")

# TRANSFORMER

# SRC_VOCAB_SIZE = en_vocab_size
# TGT_VOCAB_SIZE = de_vocab_size
# EMB_SIZE = 512
# NHEAD = 8
# FFN_HID_DIM = 512
# BATCH_SIZE = 128
# NUM_ENCODER_LAYERS = 3
# NUM_DECODER_LAYERS = 3

# transformer = lib.Seq2SeqTransformer(
#     NUM_ENCODER_LAYERS,
#     NUM_DECODER_LAYERS,
#     EMB_SIZE,
#     NHEAD,
#     SRC_VOCAB_SIZE,
#     TGT_VOCAB_SIZE,
#     FFN_HID_DIM,
# )

# for p in transformer.parameters():
#     if p.dim() > 1:
#         nn.init.xavier_uniform_(p)

# transformer = transformer.to(DEVICE)

# loss_fn = torch.nn.CrossEntropyLoss(ignore_index=de_word_padding_idx)

# optimizer = torch.optim.Adam(
#     transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
# )


# def train_epoch(en_train, de_train, model, optimizer, batch_size):
#     model.train()
#     losses = 0

#     data = list(zip(en_train, de_train))
#     random.shuffle(data)
#     en_train, de_train = zip(*data)

#     for i in range(0, len(en_train), batch_size):
#         en_batch = en_train[i : i + batch_size]
#         de_batch = de_train[i : i + batch_size]

#         # Numericalize and pad the *entire batch*
#         en_numericalized = [numericalize(text, en_sp) for text in en_batch]
#         de_numericalized = [numericalize(text, de_sp) for text in de_batch]

#         en_lengths = torch.tensor([tensor.shape[0] for tensor in en_numericalized])

#         en_input = nn.utils.rnn.pad_sequence(
#             en_numericalized, padding_value=en_word_padding_idx, batch_first=True
#         )
#         de_input = nn.utils.rnn.pad_sequence(
#             de_numericalized, padding_value=de_word_padding_idx, batch_first=True
#         )

#         src = en_input.to(device)
#         tgt = de_input.to(device)
#         en_lengths = en_lengths.to(device)

#         tgt_input = tgt[:, :-1]

#         src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = lib.create_mask(
#             src, tgt_input, en_word_padding_idx, de_word_padding_idx
#         )

#         logits = model(
#             src,
#             tgt_input,
#             src_mask,
#             tgt_mask,
#             src_padding_mask.transpose(0, 1),
#             tgt_padding_mask.transpose(0, 1),
#             src_padding_mask.transpose(0, 1),
#         )

#         optimizer.zero_grad()

#         tgt_out = tgt[:, 1:]
#         loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
#         loss.backward()

#         optimizer.step()
#         losses += loss.item()

#     return losses / len(en_train)


# from timeit import default_timer as timer

# NUM_EPOCHS = 18


# def load_data(en_filepath, de_filepath):
#     en_data = []
#     de_data = []
#     with open(en_filepath, "r", encoding="utf-8") as en_f, open(
#         de_filepath, "r", encoding="utf-8"
#     ) as de_f:
#         for en_line, de_line in zip(en_f, de_f):
#             en_data.append(en_line.strip())
#             de_data.append(de_line.strip())
#     return en_data, de_data


# en_train, de_train = load_data("multi30k_train_en.txt", "multi30k_train_de.txt")


# for epoch in range(1, NUM_EPOCHS + 1):
#     start_time = timer()
#     train_loss = train_epoch(en_train, de_train, transformer, optimizer, BATCH_SIZE)
#     end_time = timer()
#     print(
#         (
#             f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "
#             f"Epoch time = {(end_time - start_time):.3f}s"
#         )
#     )


# def greedy_decode(model, src, src_mask, max_len, start_symbol, eos_idx):
#     src = src.to(DEVICE)
#     src_mask = src_mask.to(DEVICE)

#     memory = model.encode(src, src_mask)
#     ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
#     for i in range(max_len - 1):
#         memory = memory.to(DEVICE)
#         tgt_mask = (lib.generate_square_subsequent_mask(ys.size(1)).type(torch.bool)).to(
#             DEVICE
#         )
#         out = model.decode(ys, memory, tgt_mask)
#         out = out.transpose(0, 1)
#         prob = model.generator(out[:, -1])
#         _, next_word = torch.max(prob, dim=1)  # next_word is a tensor

#         next_word = next_word[0].item()  # Get the scalar value correctly

#         ys = torch.cat(
#             [ys, torch.ones(1, 1).type_as(src.data).fill_(next_word).to(DEVICE)], dim=1
#         )
#         if next_word == eos_idx:
#             break
#     return ys


# def translate(model: torch.nn.Module, src_sentence: str):
#     model.eval()
#     src_tokenizer = en_sp
#     tgt_tokenizer = de_sp
#     with torch.no_grad():
#         src = numericalize(src_sentence, src_tokenizer).unsqueeze(0).to(device)
#         num_tokens = src.shape[1]
#         src_mask = (
#             (torch.zeros(num_tokens, num_tokens, device=device))
#             .type(torch.bool)
#             .transpose(0, 1)
#         )
#         tgt_tokens = greedy_decode(
#             model,
#             src,
#             src_mask,
#             max_len=num_tokens + 5,
#             start_symbol=tgt_tokenizer.bos_id(),
#             eos_idx=tgt_tokenizer.eos_id(),
#         ).flatten()
#         decoded_sentence = (
#             "".join([tgt_tokenizer.IdToPiece(idx.item()) for idx in tgt_tokens])
#             .replace("▁", " ")
#             .replace("<bos>", "")
#             .replace("<eos>", "")
#         )
#         return decoded_sentence

# print(translate(transformer, "Eine Gruppe von Menschen steht vor einem Iglu ."))
