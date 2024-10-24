from collections import namedtuple
import math
import warnings
import subprocess
import platform
import os
import re
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer
import torch.nn.functional as F
from torch.nn import Parameter
import torch.nn.init as init
from torch.nn.utils import skip_init
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import configargparse
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncodingPyTorch(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncodingPyTorch, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncodingPyTorch(emb_size, dropout=dropout)

    def forward(
        self,
        src: Tensor,
        trg: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor,
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

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt, src_pad_idx, tgt_pad_idx):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == src_pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == tgt_pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def context_gate_factory(
    gate_type, embeddings_size, decoder_size, attention_size, output_size
):
    """Returns the correct ContextGate class"""

    gate_types = {
        "source": SourceContextGate,
        "target": TargetContextGate,
        "both": BothContextGate,
    }

    assert gate_type in gate_types, "Not valid ContextGate type: {0}".format(gate_type)
    return gate_types[gate_type](
        embeddings_size, decoder_size, attention_size, output_size
    )


class ContextGate(nn.Module):
    """
    Context gate is a decoder module that takes as input the previous word
    embedding, the current decoder state and the attention state, and
    produces a gate.
    The gate can be used to select the input from the target side context
    (decoder state), from the source context (attention state) or both.
    """

    def __init__(self, embeddings_size, decoder_size, attention_size, output_size):
        super(ContextGate, self).__init__()
        input_size = embeddings_size + decoder_size + attention_size
        self.gate = nn.Linear(input_size, output_size, bias=True)
        self.sig = nn.Sigmoid()
        self.source_proj = nn.Linear(attention_size, output_size)
        self.target_proj = nn.Linear(embeddings_size + decoder_size, output_size)

    def forward(self, prev_emb, dec_state, attn_state):
        input_tensor = torch.cat((prev_emb, dec_state, attn_state), dim=1)
        z = self.sig(self.gate(input_tensor))
        proj_source = self.source_proj(attn_state)
        proj_target = self.target_proj(torch.cat((prev_emb, dec_state), dim=1))
        return z, proj_source, proj_target


class SourceContextGate(nn.Module):
    """Apply the context gate only to the source context"""

    def __init__(self, embeddings_size, decoder_size, attention_size, output_size):
        super(SourceContextGate, self).__init__()
        self.context_gate = ContextGate(
            embeddings_size, decoder_size, attention_size, output_size
        )
        self.tanh = nn.Tanh()

    def forward(self, prev_emb, dec_state, attn_state):
        z, source, target = self.context_gate(prev_emb, dec_state, attn_state)
        return self.tanh(target + z * source)


class TargetContextGate(nn.Module):
    """Apply the context gate only to the target context"""

    def __init__(self, embeddings_size, decoder_size, attention_size, output_size):
        super(TargetContextGate, self).__init__()
        self.context_gate = ContextGate(
            embeddings_size, decoder_size, attention_size, output_size
        )
        self.tanh = nn.Tanh()

    def forward(self, prev_emb, dec_state, attn_state):
        z, source, target = self.context_gate(prev_emb, dec_state, attn_state)
        return self.tanh(z * target + source)


class BothContextGate(nn.Module):
    """Apply the context gate to both contexts"""

    def __init__(self, embeddings_size, decoder_size, attention_size, output_size):
        super(BothContextGate, self).__init__()
        self.context_gate = ContextGate(
            embeddings_size, decoder_size, attention_size, output_size
        )
        self.tanh = nn.Tanh()

    def forward(self, prev_emb, dec_state, attn_state):
        z, source, target = self.context_gate(prev_emb, dec_state, attn_state)
        return self.tanh((1.0 - z) * target + z * source)


class StackedLSTM(nn.Module):
    """
    Our own implementation of stacked LSTM.
    Needed for the decoder, because we do input feeding.
    """

    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input_feed, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input_feed, (h_0[i], c_0[i]))
            input_feed = h_1_i
            if i + 1 != self.num_layers:
                input_feed = self.dropout(input_feed)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input_feed, (h_1, c_1)


class StackedGRU(nn.Module):
    """
    Our own implementation of stacked GRU.
    Needed for the decoder, because we do input feeding.
    """

    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input_feed, hidden):
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input_feed, hidden[0][i])
            input_feed = h_1_i
            if i + 1 != self.num_layers:
                input_feed = self.dropout(input_feed)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)
        return input_feed, (h_1,)


class RNNDecoderBase(nn.Module):
    """Base recurrent attention-based decoder class.

    Specifies the interface used by different decoder types
    and required by :class:`~eole.models.BaseModel`.

    Args:
        model_config (eole.config.DecoderConfig): full decoder config
        running_config (TrainingConfig / InferenceConfig)
    """

    def __init__(
        self,
        bidirectional_encoder,
        layers,
        rnn_type,
        dropout,
        context_gate,
        coverage_attn,
        global_attention,
        global_attention_function,
    ):
        super(RNNDecoderBase, self).__init__()

        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = layers
        self.dropout = nn.Dropout(dropout)

        # Build the RNN.
        self.rnn = self._build_rnn(
            rnn_type,
            input_size=self._input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=dropout,
        )

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = context_gate_factory(
                context_gate,
                self._input_size,
                self.hidden_size,
                self.hidden_size,
                self.hidden_size,
            )

        # Set up the standard attention.
        self._coverage = coverage_attn
        if not self.attentional:
            if self._coverage:
                raise ValueError("Cannot use coverage term with no attention.")
            self.attn = None
        else:
            self.attn = GlobalAttention(
                self.hidden_size,
                coverage=coverage_attn,
                attn_type=global_attention,
                attn_func=global_attention_function,
            )

    def init_state(self, **kwargs):
        """Initialize decoder state with last state of the encoder."""
        enc_final_hs = kwargs.pop("enc_final_hs", None)

        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                hidden = torch.cat(
                    [hidden[0 : hidden.size(0) : 2], hidden[1 : hidden.size(0) : 2]], 2
                )
            return hidden

        if isinstance(enc_final_hs, tuple):  # LSTM
            self.state["hidden"] = tuple(
                _fix_enc_hidden(enc_hid) for enc_hid in enc_final_hs
            )
        else:  # GRU
            self.state["hidden"] = (_fix_enc_hidden(enc_final_hs),)

        # Init the input feed.
        batch_size = self.state["hidden"][0].size(1)

        h_size = (batch_size, self.hidden_size)
        self.state["input_feed"] = (
            self.state["hidden"][0].data.new(*h_size).zero_().unsqueeze(0)
        )

        self.state["coverage"] = None

    def map_state(self, fn):
        self.state["hidden"] = tuple(
            fn(h.transpose(0, 1), 0).transpose(0, 1) for h in self.state["hidden"]
        )
        self.state["input_feed"] = fn(
            self.state["input_feed"].transpose(0, 1), 0
        ).transpose(0, 1)
        if self._coverage and self.state["coverage"] is not None:
            self.state["coverage"] = fn(
                self.state["coverage"].transpose(0, 1), 0
            ).transpose(0, 1)

    def detach_state(self):
        self.state["hidden"] = tuple(h.detach() for h in self.state["hidden"])
        self.state["input_feed"] = self.state["input_feed"].detach()
        if self._coverage and self.state["coverage"] is not None:
            self.state["coverage"] = self.state["coverage"].detach()

    def forward(self, emb, enc_out, src_len=None, step=None, **kwargs):
        """
        Args:
            emb (FloatTensor): input embeddings
                 ``(batch, tgt_len, dim)``.
            enc_out (FloatTensor): vectors from the encoder
                 ``(batch, src_len, hidden)``.
            src_len (LongTensor): the padded source lengths
                ``(batch,)``.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * dec_outs: output from the decoder (after attn)
              ``(batch, tgt_len, hidden)``.
            * attns: distribution over src at each tgt
              ``(batch, tgt_len, src_len)``.
        """
        dec_state, dec_outs, attns = self._run_forward_pass(
            emb, enc_out, src_len=src_len
        )

        # Update the state with the result.
        if not isinstance(dec_state, tuple):
            dec_state = (dec_state,)
        self.state["hidden"] = dec_state

        # Concatenates sequence of tensors along a new dimension.
        # NOTE: v0.3 to 0.4: dec_outs / attns[*] may not be list
        #       since stack(Variable) was allowed.
        if type(dec_outs) == list:
            dec_outs = torch.stack(dec_outs, dim=1)
            for k in attns:
                if type(attns[k]) == list:
                    attns[k] = torch.stack(attns[k])

        self.state["input_feed"] = dec_outs[:, -1, :].unsqueeze(0)
        self.state["coverage"] = None
        if "coverage" in attns:
            self.state["coverage"] = attns["coverage"][-1, :, :].unsqueeze(0)

        return dec_outs, attns

    def update_dropout(self, dropout, attention_dropout=None):
        self.dropout.p = dropout


class StdRNNDecoder(RNNDecoderBase):
    """Standard fully batched RNN decoder with attention.

    Faster implementation, uses CuDNN for implementation.
    See :class:`~eole.decoders.decoder.RNNDecoderBase` for options.


    Based around the approach from
    "Neural Machine Translation By Jointly Learning To Align and Translate"
    :cite:`Bahdanau2015`


    Implemented without input_feeding and currently with no `coverage_attn`
    """

    def __init__(
        self,
        hidden_size,
        tgt_word_vec_size,
        bidirectional_encoder,
        layers,
        rnn_type,
        dropout,
        context_gate,
        coverage_attn,
        global_attention,
        global_attention_function,
    ):
        self.hidden_size = hidden_size
        self._input_size = tgt_word_vec_size
        super(StdRNNDecoder, self).__init__(
            bidirectional_encoder,
            layers,
            rnn_type,
            dropout,
            context_gate,
            coverage_attn,
            global_attention,
            global_attention_function,
        )

    def _run_forward_pass(self, emb, enc_out, src_len=None):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.

        Args:
            emb (FloatTensor): input embeddings
                ``(batch, tgt_len, dim)``.
            enc_out (FloatTensor): output(tensor sequence) from the
                encoder RNN of size ``(batch, src_len, hidden_size)``.
            src_len (LongTensor): the source enc_out lengths.

        Returns:
            (Tensor, List[FloatTensor], Dict[str, List[FloatTensor]):

            * dec_state: final hidden state from the decoder.
            * dec_outs: an array of output of every time
              step from the decoder.
            * attns: a dictionary of different
              type of attention Tensor array of every time
              step from the decoder.
        """

        assert not self._coverage  # TODO, no support yet.

        attns = {}

        if isinstance(self.rnn, nn.GRU):
            rnn_out, dec_state = self.rnn(emb, self.state["hidden"][0])
        else:
            rnn_out, dec_state = self.rnn(emb, self.state["hidden"])

        tgt_batch, tgt_len, _ = emb.size()

        # Calculate the attention.
        if not self.attentional:
            dec_outs = rnn_out
        else:
            dec_outs, p_attn = self.attn(rnn_out, enc_out, src_len=src_len)
            attns["std"] = p_attn

        # Calculate the context gate.
        if self.context_gate is not None:
            dec_outs = self.context_gate(
                emb.view(-1, emb.size(2)),
                rnn_out.view(-1, rnn_out.size(2)),
                dec_outs.view(-1, dec_outs.size(2)),
            )
            dec_outs = dec_outs.view(tgt_batch, tgt_len, self.hidden_size)

        dec_outs = self.dropout(dec_outs)

        return dec_state, dec_outs, attns

    def _build_rnn(self, rnn_type, **kwargs):
        rnn = getattr(nn, rnn_type)(batch_first=True, **kwargs)
        return rnn


class InputFeedRNNDecoder(RNNDecoderBase):
    """Input feeding based decoder.

    See :class:`~eole.decoders.decoder.RNNDecoderBase` for options.

    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`

    """

    def __init__(
        self,
        model_config,
        running_config=None,
    ):
        self.hidden_size = model_config.hidden_size
        self._input_size = model_config.tgt_word_vec_size + self.hidden_size
        super(InputFeedRNNDecoder, self).__init__(model_config, running_config)

    def _run_forward_pass(self, emb, enc_out, src_len=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        input_feed = self.state["input_feed"].squeeze(0)

        dec_outs = []
        attns = {}
        if self.attn is not None:
            attns["std"] = []
        if self._coverage:
            attns["coverage"] = []

        assert emb.dim() == 3  # batch x len x embedding_dim

        dec_state = self.state["hidden"]

        coverage = (
            self.state["coverage"].squeeze(0)
            if self.state["coverage"] is not None
            else None
        )

        # Input feed concatenates hidden state with
        # input at every time step.
        for emb_t in emb.split(1, dim=1):
            dec_in = torch.cat([emb_t.squeeze(1), input_feed], 1)
            rnn_out, dec_state = self.rnn(dec_in, dec_state)
            if self.attentional:
                dec_out, p_attn = self.attn(rnn_out, enc_out, src_len=src_len)
                attns["std"].append(p_attn)
            else:
                dec_out = rnn_out
            if self.context_gate is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                dec_out = self.context_gate(dec_in, rnn_out, dec_out)
            dec_out = self.dropout(dec_out)
            input_feed = dec_out

            dec_outs += [dec_out]

            # Update the coverage attention.
            # attns["coverage"] is actually c^(t+1) of See et al(2017)
            # 1-index shifted
            if self._coverage:
                coverage = p_attn if coverage is None else p_attn + coverage
                attns["coverage"] += [coverage]

        return dec_state, dec_outs, attns

    def _build_rnn(self, rnn_type, input_size, hidden_size, num_layers, dropout):
        stacked_cell = StackedLSTM if rnn_type == "LSTM" else StackedGRU
        return stacked_cell(num_layers, input_size, hidden_size, dropout)

    def update_dropout(self, dropout, attention_dropout=None):
        self.dropout.p = dropout
        self.rnn.dropout.p = dropout


class GGNNAttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class GGNNPropogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """

    def __init__(self, state_dim, n_node, n_edge_types):
        super(GGNNPropogator, self).__init__()

        self.n_node = n_node
        self.n_edge_types = n_edge_types

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim), nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim), nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim), nn.LeakyReLU()
        )

    def forward(self, state_in, state_out, state_cur, edges, nodes):
        edges_in = edges[:, :, : nodes * self.n_edge_types]
        edges_out = edges[:, :, nodes * self.n_edge_types :]

        a_in = torch.bmm(edges_in, state_in)
        a_out = torch.bmm(edges_out, state_out)
        a = torch.cat((a_in, a_out, state_cur), 2)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * state_cur), 2)
        h_hat = self.tansform(joined_input)

        prop_out = (1 - z) * state_cur + z * h_hat

        return prop_out


class GGNNEncoder(nn.Module):
    """A gated graph neural network configured as an encoder.
       Based on github.com/JamesChuanggg/ggnn.pytorch.git,
       which is based on the paper "Gated Graph Sequence Neural Networks"
       by Y. Li, D. Tarlow, M. Brockschmidt, and R. Zemel.

    Args:
       rnn_type (str):
          style of recurrent unit to use, one of [LSTM]
       src_ggnn_size (int) : Size of token-to-node embedding input
       src_word_vec_size (int) : Size of token-to-node embedding output
       state_dim (int) : Number of state dimensions in nodes
       n_edge_types (int) : Number of edge types
       bidir_edges (bool): True if reverse edges should be autocreated
       n_node (int) : Max nodes in graph
       bridge_extra_node (bool): True indicates only 1st extra node
          (after token listing) should be used for decoder init.
       n_steps (int): Steps to advance graph encoder for stabilization
       src_vocab (int): Path to source vocabulary.(The ggnn uses src_vocab
            during training because the graph is built using edge information
            which requires parsing the input sequence.)
    """

    def __init__(
        self,
        rnn_type,
        src_word_vec_size,
        src_ggnn_size,
        state_dim,
        bidir_edges,
        n_edge_types,
        n_node,
        bridge_extra_node,
        n_steps,
        src_vocab,
    ):
        super(GGNNEncoder, self).__init__()

        self.src_word_vec_size = src_word_vec_size
        self.src_ggnn_size = src_ggnn_size
        self.state_dim = state_dim
        self.n_edge_types = n_edge_types
        self.n_node = n_node
        self.n_steps = n_steps
        self.bidir_edges = bidir_edges
        self.bridge_extra_node = bridge_extra_node

        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            out_fc = nn.Linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = GGNNAttrProxy(self, "in_")
        self.out_fcs = GGNNAttrProxy(self, "out_")

        # Find vocab data for tree builting
        f = open(src_vocab, "r")
        idx = 0
        self.COMMA = -1
        self.DELIMITER = -1
        self.idx2num = []
        found_n_minus_one = False
        for ln in f:
            ln = ln.strip("\n")
            ln = ln.split("\t")[0]
            if idx == 0 and ln != "<unk>":
                idx += 1
                self.idx2num.append(-1)
            if idx == 1 and ln != "<blank>":
                idx += 1
                self.idx2num.append(-1)
            if ln == ",":
                self.COMMA = idx
            if ln == "<EOT>":
                self.DELIMITER = idx
            if ln.isdigit():
                self.idx2num.append(int(ln))
                if int(ln) == n_node - 1:
                    found_n_minus_one = True
            else:
                self.idx2num.append(-1)
            idx += 1

        assert self.COMMA >= 0, "GGNN src_vocab must include ',' character"
        assert self.DELIMITER >= 0, "GGNN src_vocab must include <EOT> token"
        assert (
            found_n_minus_one
        ), "GGNN src_vocab must include node numbers for edge connections"

        # Propogation Model
        self.propogator = GGNNPropogator(self.state_dim, self.n_node, self.n_edge_types)

        self._initialization()

        # Initialize the bridge layer
        self._initialize_bridge(rnn_type, self.state_dim, 1)

        # Token embedding
        if src_ggnn_size > 0:
            self.embed = nn.Sequential(
                nn.Linear(src_ggnn_size, src_word_vec_size), nn.LeakyReLU()
            )
            assert (
                self.src_ggnn_size >= self.DELIMITER
            ), "Embedding input must be larger than vocabulary"
            assert (
                self.src_word_vec_size < self.state_dim
            ), "Embedding size must be smaller than state_dim"
        else:
            assert (
                self.DELIMITER < self.state_dim
            ), "Vocabulary too large, consider -src_ggnn_size"

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, src, src_len=None):
        """See :func:`EncoderBase.forward()`"""

        nodes = self.n_node
        batch_size = src.size()[0]
        first_extra = np.zeros(batch_size, dtype=np.int32)
        token_onehot = np.zeros(
            (
                batch_size,
                nodes,
                self.src_ggnn_size if self.src_ggnn_size > 0 else self.state_dim,
            ),
            dtype=np.int32,
        )
        edges = np.zeros(
            (batch_size, nodes, nodes * self.n_edge_types * 2), dtype=np.int32
        )
        npsrc = src[:, :, 0].cpu().data.numpy().astype(np.int32)

        # Initialize graph using formatted input sequence
        for i in range(batch_size):
            tokens_done = False
            # Number of flagged nodes defines node count for this sample
            # (Nodes can have no flags on them, but must be in 'flags' list).
            flag_node = 0
            flags_done = False
            edge = 0
            source_node = -1
            for j in range(len(npsrc)):
                token = npsrc[i][j]
                if not tokens_done:
                    if token == self.DELIMITER:
                        tokens_done = True
                        first_extra[i] = j
                    else:
                        token_onehot[i][j][token] = 1
                elif token == self.DELIMITER:
                    flag_node += 1
                    flags_done = True
                    assert flag_node <= nodes, "Too many nodes with flags"
                elif not flags_done:
                    # The total number of integers in the vocab should allow
                    # for all features and edges to be defined.
                    if token == self.COMMA:
                        flag_node = 0
                    else:
                        num = self.idx2num[token]
                        if num >= 0:
                            token_onehot[i][flag_node][num + self.DELIMITER] = 1
                        flag_node += 1
                elif token == self.COMMA:
                    edge += 1
                    assert (
                        source_node == -1
                    ), f"Error in graph edge input: {source_node} unpaired"
                    assert edge < self.n_edge_types, "Too many edge types in input"
                else:
                    num = self.idx2num[token]
                    if source_node < 0:
                        source_node = num
                    else:
                        edges[i][source_node][num + nodes * edge] = 1
                        if self.bidir_edges:
                            edges[i][num][
                                nodes * (edge + self.n_edge_types) + source_node
                            ] = 1
                        source_node = -1

        token_onehot = torch.from_numpy(token_onehot).float().to(src.device)
        if self.src_ggnn_size > 0:
            token_embed = self.embed(token_onehot)
            prop_state = torch.cat(
                (
                    token_embed,
                    torch.zeros(
                        (batch_size, nodes, self.state_dim - self.src_word_vec_size)
                    )
                    .float()
                    .to(src.device),
                ),
                2,
            )
        else:
            prop_state = token_onehot
        edges = torch.from_numpy(edges).float().to(src.device)

        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](prop_state))
                out_states.append(self.out_fcs[i](prop_state))
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, nodes * self.n_edge_types, self.state_dim)
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(-1, nodes * self.n_edge_types, self.state_dim)

            prop_state = self.propogator(
                in_states, out_states, prop_state, edges, nodes
            )

        if self.bridge_extra_node:
            # Use first extra node as only source for decoder init
            join_state = prop_state[first_extra, torch.arange(batch_size)]
        else:
            # Average all nodes to get bridge input
            join_state = prop_state.mean(0)
        join_state = torch.stack((join_state, join_state, join_state, join_state))
        join_state = (join_state, join_state)

        enc_final_hs = self._bridge(join_state)

        return prop_state, enc_final_hs, src_len

    def _initialize_bridge(self, rnn_type, hidden_size, num_layers):
        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList(
            [
                nn.Linear(self.total_hidden_dim, self.total_hidden_dim, bias=True)
                for _ in range(number_of_states)
            ]
        )

    def _bridge(self, hidden):
        """Forward hidden state through bridge."""

        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.leaky_relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple(
                [
                    bottle_hidden(layer, hidden[ix])
                    for ix, layer in enumerate(self.bridge)
                ]
            )
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs


# For command-line option parsing
class CheckSRU(configargparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super(CheckSRU, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if values == "SRU":
            check_sru_requirement(abort=True)
        # Check pass, set the args.
        setattr(namespace, self.dest, values)


# This SRU version implements its own cuda-level optimization,
# so it requires that:
# 1. `cupy` and `pynvrtc` python package installed.
# 2. pytorch is built with cuda support.
# 3. library path set: export LD_LIBRARY_PATH=<cuda lib path>.
def check_sru_requirement(abort=False):
    """
    Return True if check pass; if check fails and abort is True,
    raise an Exception, othereise return False.
    """

    # Check 1.
    try:
        if platform.system() == "Windows":
            subprocess.check_output("pip freeze | findstr cupy", shell=True)
            subprocess.check_output("pip freeze | findstr pynvrtc", shell=True)
        else:  # Unix-like systems
            subprocess.check_output("pip freeze | grep -w cupy", shell=True)
            subprocess.check_output("pip freeze | grep -w pynvrtc", shell=True)
    except subprocess.CalledProcessError:
        if not abort:
            return False
        raise AssertionError(
            "Using SRU requires 'cupy' and 'pynvrtc' " "python packages installed."
        )

    # Check 2.
    if torch.cuda.is_available() is False:
        if not abort:
            return False
        raise AssertionError("Using SRU requires pytorch built with cuda.")

    # Check 3.
    pattern = re.compile(".*cuda/lib.*")
    ld_path = os.getenv("LD_LIBRARY_PATH", "")
    if re.match(pattern, ld_path) is None:
        if not abort:
            return False
        raise AssertionError(
            "Using SRU requires setting cuda lib path, e.g. "
            "export LD_LIBRARY_PATH=/usr/local/cuda/lib64."
        )

    return True


SRU_CODE = """
extern "C" {
    __forceinline__ __device__ float sigmoidf(float x)
    {
        return 1.f / (1.f + expf(-x));
    }
    __forceinline__ __device__ float reluf(float x)
    {
        return (x > 0.f) ? x : 0.f;
    }
    __global__ void sru_fwd(const float * __restrict__ u,
                            const float * __restrict__ x,
                            const float * __restrict__ bias,
                            const float * __restrict__ init,
                            const float * __restrict__ mask_h,
                            const int len, const int batch,
                            const int d, const int k,
                            float * __restrict__ h,
                            float * __restrict__ c,
                            const int activation_type)
    {
        assert ((k == 3) || (x == NULL));
        int ncols = batch*d;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;
        int ncols_u = ncols*k;
        int ncols_x = (k == 3) ? ncols : ncols_u;
        const float bias1 = *(bias + (col%d));
        const float bias2 = *(bias + (col%d) + d);
        const float mask = (mask_h == NULL) ? 1.0 : (*(mask_h + col));
        float cur = *(init + col);
        const float *up = u + (col*k);
        const float *xp = (k == 3) ? (x + col) : (up + 3);
        float *cp = c + col;
        float *hp = h + col;
        for (int row = 0; row < len; ++row)
        {
            float g1 = sigmoidf((*(up+1))+bias1);
            float g2 = sigmoidf((*(up+2))+bias2);
            cur = (cur-(*up))*g1 + (*up);
            *cp = cur;
            float val = (activation_type == 1) ? tanh(cur) : (
                (activation_type == 2) ? reluf(cur) : cur
            );
            *hp = (val*mask-(*xp))*g2 + (*xp);
            up += ncols_u;
            xp += ncols_x;
            cp += ncols;
            hp += ncols;
        }
    }
    __global__ void sru_bwd(const float * __restrict__ u,
                            const float * __restrict__ x,
                            const float * __restrict__ bias,
                            const float * __restrict__ init,
                            const float * __restrict__ mask_h,
                            const float * __restrict__ c,
                            const float * __restrict__ grad_h,
                            const float * __restrict__ grad_last,
                            const int len,
                            const int batch, const int d, const int k,
                            float * __restrict__ grad_u,
                            float * __restrict__ grad_x,
                            float * __restrict__ grad_bias,
                            float * __restrict__ grad_init,
                            int activation_type)
    {
        assert((k == 3) || (x == NULL));
        assert((k == 3) || (grad_x == NULL));
        int ncols = batch*d;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;
        int ncols_u = ncols*k;
        int ncols_x = (k == 3) ? ncols : ncols_u;
        const float bias1 = *(bias + (col%d));
        const float bias2 = *(bias + (col%d) + d);
        const float mask = (mask_h == NULL) ? 1.0 : (*(mask_h + col));
        float gbias1 = 0;
        float gbias2 = 0;
        float cur = *(grad_last + col);
        const float *up = u + (col*k) + (len-1)*ncols_u;
        const float *xp = (k == 3) ? (x + col + (len-1)*ncols) : (up + 3);
        const float *cp = c + col + (len-1)*ncols;
        const float *ghp = grad_h + col + (len-1)*ncols;
        float *gup = grad_u + (col*k) + (len-1)*ncols_u;
        float *gxp = (k == 3) ? (grad_x + col + (len-1)*ncols) : (gup + 3);
        for (int row = len-1; row >= 0; --row)
        {
            const float g1 = sigmoidf((*(up+1))+bias1);
            const float g2 = sigmoidf((*(up+2))+bias2);
            const float c_val = (activation_type == 1) ? tanh(*cp) : (
                (activation_type == 2) ? reluf(*cp) : (*cp)
            );
            const float x_val = *xp;
            const float u_val = *up;
            const float prev_c_val = (row>0) ? (*(cp-ncols)) : (*(init+col));
            const float gh_val = *ghp;
            // h = c*g2 + x*(1-g2) = (c-x)*g2 + x
            // c = c'*g1 + g0*(1-g1) = (c'-g0)*g1 + g0
            // grad wrt x
            *gxp = gh_val*(1-g2);
            // grad wrt g2, u2 and bias2
            float gg2 = gh_val*(c_val*mask-x_val)*(g2*(1-g2));
            *(gup+2) = gg2;
            gbias2 += gg2;
            // grad wrt c
            const float tmp = (activation_type == 1) ? (g2*(1-c_val*c_val)) : (
                ((activation_type == 0) || (c_val > 0)) ? g2 : 0.f
            );
            const float gc = gh_val*mask*tmp + cur;
            // grad wrt u0
            *gup = gc*(1-g1);
            // grad wrt g1, u1, and bias1
            float gg1 = gc*(prev_c_val-u_val)*(g1*(1-g1));
            *(gup+1) = gg1;
            gbias1 += gg1;
            // grad wrt c'
            cur = gc*g1;
            up -= ncols_u;
            xp -= ncols_x;
            cp -= ncols;
            gup -= ncols_u;
            gxp -= ncols_x;
            ghp -= ncols;
        }
        *(grad_bias + col) = gbias1;
        *(grad_bias + col + ncols) = gbias2;
        *(grad_init +col) = cur;
    }
    __global__ void sru_bi_fwd(const float * __restrict__ u,
                               const float * __restrict__ x,
                               const float * __restrict__ bias,
                               const float * __restrict__ init,
                               const float * __restrict__ mask_h,
                               const int len, const int batch,
                               const int d, const int k,
                               float * __restrict__ h,
                               float * __restrict__ c,
                               const int activation_type)
    {
        assert ((k == 3) || (x == NULL));
        assert ((k == 3) || (k == 4));
        int ncols = batch*d*2;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;
        int ncols_u = ncols*k;
        int ncols_x = (k == 3) ? ncols : ncols_u;
        const float mask = (mask_h == NULL) ? 1.0 : (*(mask_h + col));
        float cur = *(init + col);
        const int d2 = d*2;
        const bool flip = (col%d2) >= d;
        const float bias1 = *(bias + (col%d2));
        const float bias2 = *(bias + (col%d2) + d2);
        const float *up = u + (col*k);
        const float *xp = (k == 3) ? (x + col) : (up + 3);
        float *cp = c + col;
        float *hp = h + col;
        if (flip) {
            up += (len-1)*ncols_u;
            xp += (len-1)*ncols_x;
            cp += (len-1)*ncols;
            hp += (len-1)*ncols;
        }
        int ncols_u_ = flip ? -ncols_u : ncols_u;
        int ncols_x_ = flip ? -ncols_x : ncols_x;
        int ncols_ = flip ? -ncols : ncols;
        for (int cnt = 0; cnt < len; ++cnt)
        {
            float g1 = sigmoidf((*(up+1))+bias1);
            float g2 = sigmoidf((*(up+2))+bias2);
            cur = (cur-(*up))*g1 + (*up);
            *cp = cur;
            float val = (activation_type == 1) ? tanh(cur) : (
                (activation_type == 2) ? reluf(cur) : cur
            );
            *hp = (val*mask-(*xp))*g2 + (*xp);
            up += ncols_u_;
            xp += ncols_x_;
            cp += ncols_;
            hp += ncols_;
        }
    }
    __global__ void sru_bi_bwd(const float * __restrict__ u,
                               const float * __restrict__ x,
                               const float * __restrict__ bias,
                               const float * __restrict__ init,
                               const float * __restrict__ mask_h,
                               const float * __restrict__ c,
                               const float * __restrict__ grad_h,
                               const float * __restrict__ grad_last,
                               const int len, const int batch,
                               const int d, const int k,
                               float * __restrict__ grad_u,
                               float * __restrict__ grad_x,
                               float * __restrict__ grad_bias,
                               float * __restrict__ grad_init,
                               int activation_type)
    {
        assert((k == 3) || (x == NULL));
        assert((k == 3) || (grad_x == NULL));
        assert((k == 3) || (k == 4));
        int ncols = batch*d*2;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;
        int ncols_u = ncols*k;
        int ncols_x = (k == 3) ? ncols : ncols_u;
        const float mask = (mask_h == NULL) ? 1.0 : (*(mask_h + col));
        float gbias1 = 0;
        float gbias2 = 0;
        float cur = *(grad_last + col);
        const int d2 = d*2;
        const bool flip = ((col%d2) >= d);
        const float bias1 = *(bias + (col%d2));
        const float bias2 = *(bias + (col%d2) + d2);
        const float *up = u + (col*k);
        const float *xp = (k == 3) ? (x + col) : (up + 3);
        const float *cp = c + col;
        const float *ghp = grad_h + col;
        float *gup = grad_u + (col*k);
        float *gxp = (k == 3) ? (grad_x + col) : (gup + 3);
        if (!flip) {
            up += (len-1)*ncols_u;
            xp += (len-1)*ncols_x;
            cp += (len-1)*ncols;
            ghp += (len-1)*ncols;
            gup += (len-1)*ncols_u;
            gxp += (len-1)*ncols_x;
        }
        int ncols_u_ = flip ? -ncols_u : ncols_u;
        int ncols_x_ = flip ? -ncols_x : ncols_x;
        int ncols_ = flip ? -ncols : ncols;
        for (int cnt = 0; cnt < len; ++cnt)
        {
            const float g1 = sigmoidf((*(up+1))+bias1);
            const float g2 = sigmoidf((*(up+2))+bias2);
            const float c_val = (activation_type == 1) ? tanh(*cp) : (
                (activation_type == 2) ? reluf(*cp) : (*cp)
            );
            const float x_val = *xp;
            const float u_val = *up;
            const float prev_c_val = (cnt<len-1)?(*(cp-ncols_)):(*(init+col));
            const float gh_val = *ghp;
            // h = c*g2 + x*(1-g2) = (c-x)*g2 + x
            // c = c'*g1 + g0*(1-g1) = (c'-g0)*g1 + g0
            // grad wrt x
            *gxp = gh_val*(1-g2);
            // grad wrt g2, u2 and bias2
            float gg2 = gh_val*(c_val*mask-x_val)*(g2*(1-g2));
            *(gup+2) = gg2;
            gbias2 += gg2;
            // grad wrt c
            const float tmp = (activation_type == 1) ? (g2*(1-c_val*c_val)) : (
                ((activation_type == 0) || (c_val > 0)) ? g2 : 0.f
            );
            const float gc = gh_val*mask*tmp + cur;
            // grad wrt u0
            *gup = gc*(1-g1);
            // grad wrt g1, u1, and bias1
            float gg1 = gc*(prev_c_val-u_val)*(g1*(1-g1));
            *(gup+1) = gg1;
            gbias1 += gg1;
            // grad wrt c'
            cur = gc*g1;
            up -= ncols_u_;
            xp -= ncols_x_;
            cp -= ncols_;
            gup -= ncols_u_;
            gxp -= ncols_x_;
            ghp -= ncols_;
        }
        *(grad_bias + col) = gbias1;
        *(grad_bias + col + ncols) = gbias2;
        *(grad_init +col) = cur;
    }
}
"""
SRU_FWD_FUNC, SRU_BWD_FUNC = None, None
SRU_BiFWD_FUNC, SRU_BiBWD_FUNC = None, None
SRU_STREAM = None


def load_sru_mod():
    global SRU_FWD_FUNC, SRU_BWD_FUNC, SRU_BiFWD_FUNC, SRU_BiBWD_FUNC
    global SRU_STREAM
    if check_sru_requirement():
        from cupy.cuda import function
        from pynvrtc.compiler import Program

        # This sets up device to use.
        device = torch.device("cuda")
        tmp_ = torch.rand(1, 1).to(device)

        sru_prog = Program(SRU_CODE.encode("utf-8"), "sru_prog.cu".encode("utf-8"))
        sru_ptx = sru_prog.compile()
        sru_mod = function.Module()
        sru_mod.load(bytes(sru_ptx.encode()))

        SRU_FWD_FUNC = sru_mod.get_function("sru_fwd")
        SRU_BWD_FUNC = sru_mod.get_function("sru_bwd")
        SRU_BiFWD_FUNC = sru_mod.get_function("sru_bi_fwd")
        SRU_BiBWD_FUNC = sru_mod.get_function("sru_bi_bwd")

        stream = namedtuple("Stream", ["ptr"])
        SRU_STREAM = stream(ptr=torch.cuda.current_stream().cuda_stream)


class SRU_Compute(Function):
    def __init__(self, activation_type, d_out, bidirectional=False):
        SRU_Compute.maybe_load_sru_mod()
        super(SRU_Compute, self).__init__()
        self.activation_type = activation_type
        self.d_out = d_out
        self.bidirectional = bidirectional

    @staticmethod
    def maybe_load_sru_mod():
        global SRU_FWD_FUNC

        if SRU_FWD_FUNC is None:
            load_sru_mod()

    @custom_fwd
    def forward(self, u, x, bias, init=None, mask_h=None):
        bidir = 2 if self.bidirectional else 1
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        d = self.d_out
        k = u.size(-1) // d
        k_ = k // 2 if self.bidirectional else k
        ncols = batch * d * bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols - 1) // thread_per_block + 1

        init_ = x.new(ncols).zero_() if init is None else init
        size = (length, batch, d * bidir) if x.dim() == 3 else (batch, d * bidir)
        c = x.new(*size)
        h = x.new(*size)

        FUNC = SRU_FWD_FUNC if not self.bidirectional else SRU_BiFWD_FUNC
        FUNC(
            args=[
                u.contiguous().data_ptr(),
                x.contiguous().data_ptr() if k_ == 3 else 0,
                bias.data_ptr(),
                init_.contiguous().data_ptr(),
                mask_h.data_ptr() if mask_h is not None else 0,
                length,
                batch,
                d,
                k_,
                h.data_ptr(),
                c.data_ptr(),
                self.activation_type,
            ],
            block=(thread_per_block, 1, 1),
            grid=(num_block, 1, 1),
            stream=SRU_STREAM,
        )

        self.save_for_backward(u, x, bias, init, mask_h)
        self.intermediate = c
        if x.dim() == 2:
            last_hidden = c
        elif self.bidirectional:
            # -> directions x batch x dim
            last_hidden = torch.stack((c[-1, :, :d], c[0, :, d:]))
        else:
            last_hidden = c[-1]
        return h, last_hidden

    @custom_bwd
    def backward(self, grad_h, grad_last):
        if self.bidirectional:
            grad_last = torch.cat((grad_last[0], grad_last[1]), 1)
        bidir = 2 if self.bidirectional else 1
        u, x, bias, init, mask_h = self.saved_tensors
        c = self.intermediate
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        d = self.d_out
        k = u.size(-1) // d
        k_ = k // 2 if self.bidirectional else k
        ncols = batch * d * bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols - 1) // thread_per_block + 1

        init_ = x.new(ncols).zero_() if init is None else init
        grad_u = u.new(*u.size())
        grad_bias = x.new(2, batch, d * bidir)
        grad_init = x.new(batch, d * bidir)

        # For DEBUG
        # size = (length, batch, x.size(-1)) \
        #         if x.dim() == 3 else (batch, x.size(-1))
        # grad_x = x.new(*x.size()) if k_ == 3 else x.new(*size).zero_()

        # Normal use
        grad_x = x.new(*x.size()) if k_ == 3 else None

        FUNC = SRU_BWD_FUNC if not self.bidirectional else SRU_BiBWD_FUNC
        FUNC(
            args=[
                u.contiguous().data_ptr(),
                x.contiguous().data_ptr() if k_ == 3 else 0,
                bias.data_ptr(),
                init_.contiguous().data_ptr(),
                mask_h.data_ptr() if mask_h is not None else 0,
                c.data_ptr(),
                grad_h.contiguous().data_ptr(),
                grad_last.contiguous().data_ptr(),
                length,
                batch,
                d,
                k_,
                grad_u.data_ptr(),
                grad_x.data_ptr() if k_ == 3 else 0,
                grad_bias.data_ptr(),
                grad_init.data_ptr(),
                self.activation_type,
            ],
            block=(thread_per_block, 1, 1),
            grid=(num_block, 1, 1),
            stream=SRU_STREAM,
        )
        return grad_u, grad_x, grad_bias.sum(1).view(-1), grad_init, None


class SRUCell(nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        dropout=0,
        rnn_dropout=0,
        bidirectional=False,
        use_tanh=1,
        use_relu=0,
    ):
        super(SRUCell, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.rnn_dropout = rnn_dropout
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.activation_type = 2 if use_relu else (1 if use_tanh else 0)

        out_size = n_out * 2 if bidirectional else n_out
        k = 4 if n_in != out_size else 3
        self.size_per_dir = n_out * k
        self.weight = nn.Parameter(
            torch.Tensor(
                n_in, self.size_per_dir * 2 if bidirectional else self.size_per_dir
            )
        )
        self.bias = nn.Parameter(
            torch.Tensor(n_out * 4 if bidirectional else n_out * 2)
        )
        self.init_weight()

    def init_weight(self):
        val_range = (3.0 / self.n_in) ** 0.5
        self.weight.data.uniform_(-val_range, val_range)
        self.bias.data.zero_()

    def set_bias(self, bias_val=0):
        n_out = self.n_out
        if self.bidirectional:
            self.bias.data[n_out * 2 :].zero_().add_(bias_val)
        else:
            self.bias.data[n_out:].zero_().add_(bias_val)

    def forward(self, input, c0=None):
        assert input.dim() == 2 or input.dim() == 3
        n_in, n_out = self.n_in, self.n_out
        batch = input.size(-2)
        if c0 is None:
            c0 = input.data.new(
                batch, n_out if not self.bidirectional else n_out * 2
            ).zero_()

        if self.training and (self.rnn_dropout > 0):
            mask = self.get_dropout_mask_((batch, n_in), self.rnn_dropout)
            x = input * mask.expand_as(input)
        else:
            x = input

        x_2d = x if x.dim() == 2 else x.contiguous().view(-1, n_in)
        u = x_2d.mm(self.weight)

        if self.training and (self.dropout > 0):
            bidir = 2 if self.bidirectional else 1
            mask_h = self.get_dropout_mask_((batch, n_out * bidir), self.dropout)
            h, c = SRU_Compute(self.activation_type, n_out, self.bidirectional)(
                u, input, self.bias, c0, mask_h
            )
        else:
            h, c = SRU_Compute(self.activation_type, n_out, self.bidirectional)(
                u, input, self.bias, c0
            )

        return h, c

    def get_dropout_mask_(self, size, p):
        w = self.weight.data
        return w.new(*size).bernoulli_(1 - p).div_(1 - p)


class SRU(nn.Module):
    """
    Implementation of "Training RNNs as Fast as CNNs"
    :cite:`DBLP:journals/corr/abs-1709-02755`

    TODO: turn to pytorch's implementation when it is available.

    This implementation is adpoted from the author of the paper:
    https://github.com/taolei87/sru/blob/master/cuda_functional.py.

    Args:
      input_size (int): input to model
      hidden_size (int): hidden dimension
      num_layers (int): number of layers
      dropout (float): dropout to use (stacked)
      rnn_dropout (float): dropout to use (recurrent)
      bidirectional (bool): bidirectional
      use_tanh (bool): activation
      use_relu (bool): activation
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=2,
        dropout=0,
        rnn_dropout=0,
        bidirectional=False,
        use_tanh=1,
        use_relu=0,
    ):
        # An entry check here, will catch on train side and translate side
        # if requirements are not satisfied.
        check_sru_requirement(abort=True)
        super(SRU, self).__init__()
        self.n_in = input_size
        self.n_out = hidden_size
        self.depth = num_layers
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout
        self.rnn_lst = nn.ModuleList()
        self.bidirectional = bidirectional
        self.out_size = hidden_size * 2 if bidirectional else hidden_size

        for i in range(num_layers):
            sru_cell = SRUCell(
                n_in=self.n_in if i == 0 else self.out_size,
                n_out=self.n_out,
                dropout=dropout if i + 1 != num_layers else 0,
                rnn_dropout=rnn_dropout,
                bidirectional=bidirectional,
                use_tanh=use_tanh,
                use_relu=use_relu,
            )
            self.rnn_lst.append(sru_cell)

    def set_bias(self, bias_val=0):
        for l in self.rnn_lst:
            l.set_bias(bias_val)

    def forward(self, input, c0=None, return_hidden=True):
        assert input.dim() == 3  # (len, batch, n_in)
        dir_ = 2 if self.bidirectional else 1
        if c0 is None:
            zeros = input.data.new(input.size(1), self.n_out * dir_).zero_()
            c0 = [zeros for i in range(self.depth)]
        else:
            if isinstance(c0, tuple):
                # RNNDecoderState wraps hidden as a tuple.
                c0 = c0[0]
            assert c0.dim() == 3  # (depth, batch, dir_*n_out)
            c0 = [h.squeeze(0) for h in c0.chunk(self.depth, 0)]

        prevx = input
        lstc = []
        for i, rnn in enumerate(self.rnn_lst):
            h, c = rnn(prevx, c0[i])
            prevx = h
            lstc.append(c)

        if self.bidirectional:
            # fh -> (layers*directions) x batch x dim
            fh = torch.cat(lstc)
        else:
            fh = torch.stack(lstc)

        if return_hidden:
            return prevx, fh
        else:
            return prevx


def rnn_factory(rnn_type, **kwargs):
    """rnn factory, Use pytorch version when available."""
    no_pack_padded_seq = False
    if rnn_type == "SRU":
        # SRU doesn't support PackedSequence.
        no_pack_padded_seq = True
        rnn = sru.SRU(batch_first=True, **kwargs)
    else:
        rnn = getattr(nn, rnn_type)(batch_first=True, **kwargs)
    return rnn, no_pack_padded_seq


class RNNEncoder(nn.Module):
    """A generic recurrent neural network encoder.

    Args:
       rnn_type (str):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :class:`torch.nn.Dropout`
       embeddings (onmt.modules.Embeddings): embedding module to use
    """

    def __init__(
        self,
        rnn_type,
        bidirectional,
        num_layers,
        hidden_size,
        dropout=0.0,
        embeddings=None,
        use_bridge=False,
    ):
        super(RNNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings

        self.rnn, self.no_pack_padded_seq = rnn_factory(
            rnn_type,
            input_size=embeddings.embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type, hidden_size, num_layers)

    def forward(self, src, src_len=None):
        """See :func:`EncoderBase.forward()`"""

        emb = self.embeddings(src)

        packed_emb = emb
        if src_len is not None and not self.no_pack_padded_seq:
            # src lengths data is wrapped inside a Tensor.
            src_len_list = src_len.view(-1).tolist()
            packed_emb = pack(emb, src_len_list, batch_first=True, enforce_sorted=False)

        enc_out, enc_final_hs = self.rnn(packed_emb)

        if src_len is not None and not self.no_pack_padded_seq:
            enc_out = unpack(enc_out, batch_first=True)[0]

        if self.use_bridge:
            enc_final_hs = self._bridge(enc_final_hs)

        return enc_out, enc_final_hs, src_len

    def _initialize_bridge(self, rnn_type, hidden_size, num_layers):
        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList(
            [
                nn.Linear(self.total_hidden_dim, self.total_hidden_dim, bias=True)
                for _ in range(number_of_states)
            ]
        )

    def _bridge(self, hidden):
        """Forward hidden state through bridge.
        final hidden state ``(num_layers x dir, batch, hidden_size)``
        """

        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            states = states.permute(1, 0, 2).contiguous()
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            result = F.relu(result).view(size)
            return result.permute(1, 0, 2).contiguous()

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple(
                [
                    bottle_hidden(layer, hidden[ix])
                    for ix, layer in enumerate(self.bridge)
                ]
            )
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs

    def update_dropout(self, dropout, attention_dropout=None):
        self.rnn.dropout = dropout


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    max_len = max_len or lengths.max()
    return torch.arange(0, max_len, device=lengths.device) >= lengths.unsqueeze(1)


def _make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def _threshold_and_support(input, dim=0):
    """Sparsemax building block: compute the threshold

    Args:
        input: any dimension
        dim: dimension along which to apply the sparsemax

    Returns:
        the threshold value
    """

    input_srt, _ = torch.sort(input, descending=True, dim=dim)
    input_cumsum = input_srt.cumsum(dim) - 1
    rhos = _make_ix_like(input, dim)
    support = rhos * input_srt > input_cumsum

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = input_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(input.dtype)
    return tau, support_size


class SparsemaxFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, dim=0):
        """sparsemax: normalizing sparse transform (a la softmax)

        Parameters:
            input (Tensor): any shape
            dim: dimension along which to apply sparsemax

        Returns:
            output (Tensor): same shape as input
        """
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val  # same numerical stability trick as for softmax
        tau, supp_size = _threshold_and_support(input, dim=dim)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None


sparsemax = SparsemaxFunction.apply


class Sparsemax(nn.Module):
    def __init__(self, dim=0):
        self.dim = dim
        super(Sparsemax, self).__init__()

    def forward(self, input):
        return sparsemax(input, self.dim)


class LogSparsemax(nn.Module):
    def __init__(self, dim=0):
        self.dim = dim
        super(LogSparsemax, self).__init__()

    def forward(self, input):
        return torch.log(sparsemax(input, self.dim))


SCALE_WEIGHT = 0.5**0.5


def shape_transform(x):
    """Tranform the size of the tensors to fit for conv input."""
    return torch.unsqueeze(torch.transpose(x, 1, 2), 3)


class GatedConv(nn.Module):
    """Gated convolution for CNN class"""

    def __init__(self, input_size, width=3, dropout=0.2, nopad=False):
        super(GatedConv, self).__init__()
        self.conv = WeightNormConv2d(
            input_size,
            2 * input_size,
            kernel_size=(width, 1),
            stride=(1, 1),
            padding=(width // 2 * (1 - nopad), 0),
        )
        # this param init is overridden by model_builder, useless then.
        init.xavier_uniform_(self.conv.weight, gain=(4 * (1 - dropout)) ** 0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_var):
        x_var = self.dropout(x_var)
        x_var = self.conv(x_var)
        out, gate = x_var.split(int(x_var.size(1) / 2), 1)
        out = out * torch.sigmoid(gate)
        return out


class StackedCNN(nn.Module):
    """Stacked CNN class"""

    def __init__(self, num_layers, input_size, cnn_kernel_width=3, dropout=0.2):
        super(StackedCNN, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GatedConv(input_size, cnn_kernel_width, dropout))

    def forward(self, x):
        for conv in self.layers:
            x = x + conv(x)
            x *= SCALE_WEIGHT
        return x


# This class is mainly used by decoder.py for RNNs but also
# by the CNN / transformer decoder when copy attention is used
# CNN has its own attention mechanism ConvMultiStepAttention
# Transformer has its own MultiHeadedAttention


class GlobalAttention(nn.Module):
    r"""
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.

    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.


    .. mermaid::

       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G

    All models compute the output as
    :math:`c = \sum_{j=1}^{\text{SeqLength}} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].

    However they
    differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`\text{score}(H_j,q) = H_j^T q`
       * general: :math:`\text{score}(H_j, q) = H_j^T W_a q`


    * Bahdanau Attention (mlp):
       * :math:`\text{score}(H_j, q) = v_a^T \text{tanh}(W_a q + U_a h_j)`


    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]
       attn_func (str): attention function to use, options [softmax,sparsemax]

    """

    def __init__(self, dim, coverage=False, attn_type="dot", attn_func="softmax"):
        super(GlobalAttention, self).__init__()

        self.dim = dim
        assert attn_type in [
            "dot",
            "general",
            "mlp",
        ], "Please select a valid attention type (got {:s}).".format(attn_type)
        self.attn_type = attn_type
        assert attn_func in [
            "softmax",
            "sparsemax",
        ], "Please select a valid attention function."
        self.attn_func = attn_func

        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)
        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)

        if coverage:
            self.linear_cover = nn.Linear(1, dim, bias=False)

    def score(self, h_t, h_s):
        """
        Args:
          h_t (FloatTensor): sequence of queries ``(batch, tgt_len, dim)``
          h_s (FloatTensor): sequence of sources ``(batch, src_len, dim``

        Returns:
          FloatTensor: raw attention scores (unnormalized) for each src index
            ``(batch, tgt_len, src_len)``
        """
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t = self.linear_in(h_t)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            dim = self.dim
            wq = self.linear_query(h_t)
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s.contiguous())
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = torch.tanh(wq + uh)

            return self.v(wquh).view(tgt_batch, tgt_len, src_len)

    def forward(self, src, enc_out, src_len=None, coverage=None):
        """

        Args:
          src (FloatTensor): query vectors ``(batch, tgt_len, dim)``
          enc_out (FloatTensor): encoder out vectors ``(batch, src_len, dim)``
          src_len (LongTensor): source context lengths ``(batch,)``
          coverage (FloatTensor): None (not supported yet)

        Returns:
          (FloatTensor, FloatTensor):

          * Computed vector ``(batch, tgt_len, dim)``
          * Attention distribtutions for each query
            ``(batch, tgt_len, src_len)``
        """

        # one step input
        if src.dim() == 2:
            one_step = True
            src = src.unsqueeze(1)
        else:
            one_step = False

        batch, src_l, dim = enc_out.size()
        batch_, target_l, dim_ = src.size()
        if coverage is not None:
            batch_, src_l_ = coverage.size()

        if coverage is not None:
            cover = coverage.view(-1).unsqueeze(1)
            enc_out += self.linear_cover(cover).view_as(enc_out)
            enc_out = torch.tanh(enc_out)

        # compute attention scores, as in Luong et al.
        align = self.score(src, enc_out)

        if src_len is not None:
            mask = ~sequence_mask(src_len, max_len=align.size(-1))
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.masked_fill_(~mask, -float("inf"))

        # Softmax or sparsemax to normalize attention weights
        if self.attn_func == "softmax":
            align_vectors = F.softmax(align.view(batch * target_l, src_l), -1)
        else:
            align_vectors = sparsemax(align.view(batch * target_l, src_l), -1)
        align_vectors = align_vectors.view(batch, target_l, src_l)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, enc_out)

        # concatenate
        concat_c = torch.cat([c, src], 2).view(batch * target_l, dim * 2)
        attn_h = self.linear_out(concat_c).view(batch, target_l, dim)
        if self.attn_type in ["general", "dot"]:
            attn_h = torch.tanh(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)

        return attn_h, align_vectors


def seq_linear(linear, x):
    """linear transform for 3-d tensor"""
    batch, hidden_size, length, _ = x.size()
    h = linear(torch.transpose(x, 1, 2).contiguous().view(batch * length, hidden_size))
    return torch.transpose(h.view(batch, length, hidden_size, 1), 1, 2)


class ConvMultiStepAttention(nn.Module):
    """
    Conv attention takes a key matrix, a value matrix and a query vector.
    Attention weight is calculated by key matrix with the query vector
    and sum on the value matrix. And the same operation is applied
    in each decode conv layer.
    """

    def __init__(self, input_size):
        super(ConvMultiStepAttention, self).__init__()
        self.linear_in = nn.Linear(input_size, input_size)
        self.mask = None

    def apply_mask(self, mask):
        """Apply mask"""
        self.mask = mask

    def forward(
        self, base_target_emb, input_from_dec, encoder_out_top, encoder_out_combine
    ):
        """
        Args:
            base_target_emb: target emb tensor
                ``(batch, channel, height, width)``
            input_from_dec: output of dec conv
                ``(batch, channel, height, width)``
            encoder_out_top: the key matrix for calc of attention weight,
                which is the top output of encode conv
            encoder_out_combine:
                the value matrix for the attention-weighted sum,
                which is the combination of base emb and top output of encode
        """

        preatt = seq_linear(self.linear_in, input_from_dec)
        target = (base_target_emb + preatt) * SCALE_WEIGHT
        target = torch.squeeze(target, 3)
        target = torch.transpose(target, 1, 2)

        # pre_attn = torch.bmm(target, encoder_out_top)
        pre_attn = torch.bmm(
            target, encoder_out_top.transpose(1, 2)
        )  # Transpose encoder_out_top (GEMINI)

        if self.mask is not None:
            pre_attn.data.masked_fill_(self.mask, -float("inf"))

        attn = F.softmax(pre_attn, dim=2)

        context_output = torch.bmm(attn, torch.transpose(encoder_out_combine, 1, 2))
        context_output = torch.transpose(torch.unsqueeze(context_output, 3), 1, 2)
        return context_output, attn


class Elementwise(nn.ModuleList):
    def __init__(self, merge=None, *args):
        assert merge in [None, "first", "concat", "sum", "mlp"]
        self.merge = merge
        super(Elementwise, self).__init__(*args)

    def forward(self, emb):
        if emb.dim() == 3:  # Check if it's 3D before splitting
            emb_ = [feat.squeeze(2) for feat in emb.split(1, dim=2)]
        else:
            emb_ = [
                emb
            ]  # If no features are present, 'emb' will be the output from word embedding which is 2D, then we pack the 2D tensor into a list.

        emb_out = []
        for f, x in zip(self, emb_):
            emb_out.append(f(x))

        if self.merge == "first":
            return emb_out[0]
        elif self.merge == "concat" or self.merge == "mlp":
            return torch.cat(
                emb_out, 2 if emb.dim() == 3 else 1
            )  # If 'emb' is 2D then concatenate along dimension 1
        elif self.merge == "sum":
            return sum(emb_out)
        else:
            return emb_out


class SequenceTooLongError(Exception):
    pass


class PositionalEncodingOpenNMT(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dim (int): embedding size
    """

    def __init__(self, dim, enc_type, max_len=5000):
        if dim % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(dim)
            )
        if enc_type == "SinusoidalInterleaved":
            pe = torch.zeros(max_len, dim)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(
                (
                    torch.arange(0, dim, 2, dtype=torch.float)
                    * -(math.log(10000.0) / dim)
                )
            )
            pe[:, 0::2] = torch.sin(position.float() * div_term)
            pe[:, 1::2] = torch.cos(position.float() * div_term)
        elif enc_type == "SinusoidalConcat":
            half_dim = dim // 2
            pe = math.log(10000) / (half_dim - 1)
            pe = torch.exp(torch.arange(half_dim, dtype=torch.float) * -pe)
            pe = torch.arange(max_len, dtype=torch.float).unsqueeze(1) * pe.unsqueeze(0)
            pe = torch.cat([torch.sin(pe), torch.cos(pe)], dim=1).view(max_len, -1)
        else:
            raise ValueError(
                "Choice of Position encoding is SinusoidalInterleaved or"
                " SinusoidalConcat."
            )
        pe = pe.unsqueeze(1)  # we keep pe (len x batch x dim) for back comp
        super(PositionalEncodingOpenNMT, self).__init__()
        self.register_buffer("pe", pe)
        self.dim = dim

    def forward(self, emb, step=None):
        """Embed inputs.

        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(batch_size, seq_len, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """
        pe = self.pe.transpose(0, 1)  # (batch x len x dim)
        emb = emb * math.sqrt(self.dim)
        step = step or 0
        if pe.size(1) < step + emb.size(1):
            raise SequenceTooLongError(
                f"Sequence is {emb.size(1) + step} but PositionalEncodingOpenNMT is"
                f" limited to {self.pe.size(1)}. See max_len argument."
            )
        emb = emb + pe[:, step : emb.size(1) + step, :]

        return emb


class Embeddings(nn.Module):
    """Words embeddings for encoder/decoder.

    Additionally includes ability to add sparse input features
    based on "Linguistic Input Features Improve Neural Machine Translation"
    :cite:`sennrich2016linguistic`.


    .. mermaid::

       graph LR
          A[Input]
          C[Feature 1 Lookup]
          A-->B[Word Lookup]
          A-->C
          A-->D[Feature N Lookup]
          B-->E[MLP/Concat]
          C-->E
          D-->E
          E-->F[Output]

    Args:
        word_vec_size (int): size of the dictionary of embeddings.
        word_vocab_size (int): size of dictionary of embeddings for words.
        word_padding_idx (int): padding index for words in the embeddings.
        position_encoding (bool): see :class:`~onmt.modules.PositionalEncodingOpenNMT`
        feat_merge (string): merge action for the features embeddings:
            concat, sum or mlp.
        feat_vec_exponent (float): when using `-feat_merge concat`, feature
            embedding size is N^feat_dim_exponent, where N is the
            number of values the feature takes.
        feat_vec_size (int): embedding dimension for features when using
            `-feat_merge mlp`
        feat_padding_idx (List[int]): padding index for a list of features
                                   in the embeddings.
        feat_vocab_sizes (List[int], optional): list of size of dictionary
            of embeddings for each feature.
        dropout (float): dropout probability.
        sparse (bool): sparse embbedings default False
        freeze_word_vecs (bool): freeze weights of word vectors.
    """

    def __init__(
        self,
        word_vec_size,
        word_vocab_size,
        word_padding_idx,
        position_encoding=False,
        position_encoding_type="SinusoidalInterleaved",
        feat_merge="concat",
        feat_vec_exponent=0.7,
        feat_vec_size=-1,
        feat_padding_idx=[],
        feat_vocab_sizes=[],
        dropout=0,
        sparse=False,
        freeze_word_vecs=False,
    ):
        self._validate_args(
            feat_merge,
            feat_vocab_sizes,
            feat_vec_exponent,
            feat_vec_size,
            feat_padding_idx,
        )

        if feat_padding_idx is None:
            feat_padding_idx = []
        self.word_padding_idx = word_padding_idx

        self.word_vec_size = word_vec_size

        # Dimensions and padding for constructing the word embedding matrix
        vocab_sizes = [word_vocab_size]
        emb_dims = [word_vec_size]
        pad_indices = [word_padding_idx]

        # Dimensions and padding for feature embedding matrices
        # (these have no effect if feat_vocab_sizes is empty)
        if feat_merge == "sum":
            feat_dims = [word_vec_size] * len(feat_vocab_sizes)
        elif feat_vec_size > 0:
            feat_dims = [feat_vec_size] * len(feat_vocab_sizes)
        else:
            feat_dims = [int(vocab**feat_vec_exponent) for vocab in feat_vocab_sizes]
        vocab_sizes.extend(feat_vocab_sizes)
        emb_dims.extend(feat_dims)
        pad_indices.extend(feat_padding_idx)

        # The embedding matrix look-up tables. The first look-up table
        # is for words. Subsequent ones are for features, if any exist.
        emb_params = zip(vocab_sizes, emb_dims, pad_indices)
        embeddings = [
            skip_init(
                nn.Embedding,
                num_embeddings=vocab,
                embedding_dim=dim,
                padding_idx=pad,
                sparse=sparse,
            )
            for vocab, dim, pad in emb_params
        ]
        emb_luts = Elementwise(feat_merge, embeddings)

        # The final output size of word + feature vectors. This can vary
        # from the word vector size if and only if features are defined.
        # This is the attribute you should access if you need to know
        # how big your embeddings are going to be.
        self.embedding_size = sum(emb_dims) if feat_merge == "concat" else word_vec_size

        # The sequence of operations that converts the input sequence
        # into a sequence of embeddings. At minimum this consists of
        # looking up the embeddings for each word and feature in the
        # input. Model parameters may require the sequence to contain
        # additional operations as well.
        super(Embeddings, self).__init__()
        self.make_embedding = nn.Sequential()
        self.make_embedding.add_module("emb_luts", emb_luts)

        if feat_merge == "mlp" and len(feat_vocab_sizes) > 0:
            in_dim = sum(emb_dims)
            mlp = nn.Sequential(nn.Linear(in_dim, word_vec_size), nn.ReLU())
            self.make_embedding.add_module("mlp", mlp)

        self.position_encoding = position_encoding
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_p = dropout

        if self.position_encoding:
            pe = PositionalEncodingOpenNMT(self.embedding_size, position_encoding_type)
            self.make_embedding.add_module("pe", pe)

        if freeze_word_vecs:
            self.word_lut.weight.requires_grad = False

    def _validate_args(
        self,
        feat_merge,
        feat_vocab_sizes,
        feat_vec_exponent,
        feat_vec_size,
        feat_padding_idx,
    ):
        if feat_merge == "sum":
            # features must use word_vec_size
            if feat_vec_exponent != 0.7:
                warnings.warn(
                    "Merging with sum, but got non-default "
                    "feat_vec_exponent. It will be unused."
                )
            if feat_vec_size != -1:
                warnings.warn(
                    "Merging with sum, but got non-default "
                    "feat_vec_size. It will be unused."
                )
        elif feat_vec_size > 0:
            # features will use feat_vec_size
            if feat_vec_exponent != -1:
                warnings.warn(
                    "Not merging with sum and positive "
                    "feat_vec_size, but got non-default "
                    "feat_vec_exponent. It will be unused."
                )
        else:
            if feat_vec_exponent <= 0:
                raise ValueError(
                    "Using feat_vec_exponent to determine "
                    "feature vec size, but got feat_vec_exponent "
                    "less than or equal to 0."
                )
        n_feats = len(feat_vocab_sizes)
        if n_feats != len(feat_padding_idx):
            raise ValueError(
                "Got unequal number of feat_vocab_sizes and "
                "feat_padding_idx ({:d} != {:d})".format(n_feats, len(feat_padding_idx))
            )

    @property
    def word_lut(self):
        """Word look-up table."""
        return self.make_embedding[0][0]

    @property
    def emb_luts(self):
        """Embedding look-up table."""
        return self.make_embedding[0]

    def load_pretrained_vectors(self, emb_file):
        """Load in pretrained embeddings.

        Args:
          emb_file (str) : path to torch serialized embeddings
        """

        if emb_file:
            pretrained = torch.load(emb_file)
            pretrained_vec_size = pretrained.size(1)
            if self.word_vec_size > pretrained_vec_size:
                self.word_lut.weight.data[:, :pretrained_vec_size] = pretrained
            elif self.word_vec_size < pretrained_vec_size:
                self.word_lut.weight.data.copy_(pretrained[:, : self.word_vec_size])
            else:
                self.word_lut.weight.data.copy_(pretrained)

    def forward(self, source, step=None):
        """Computes the embeddings for words and features.

        Args:
            source (LongTensor): index tensor ``(batch, len, nfeat)``

        Returns:
            FloatTensor: Word embeddings ``(batch, len, embedding_size)``
        """

        if self.position_encoding:
            for i, module in enumerate(self.make_embedding._modules.values()):
                if i == len(self.make_embedding._modules.values()) - 1:
                    source = module(source, step=step)
                else:
                    source = module(source)
        else:
            source = self.make_embedding(source)

        if self.dropout_p > 0:
            return self.dropout(source)
        else:
            return source

    def update_dropout(self, dropout):
        self.dropout.p = dropout


def get_var_maybe_avg(namespace, var_name, training, polyak_decay):
    """utility for retrieving polyak averaged params
    Update average
    """
    v = getattr(namespace, var_name)
    v_avg = getattr(namespace, var_name + "_avg")
    v_avg -= (1 - polyak_decay) * (v_avg - v.data)

    if training:
        return v
    else:
        return v_avg


def get_vars_maybe_avg(namespace, var_names, training, polyak_decay):
    """utility for retrieving polyak averaged params"""
    vars = []
    for vn in var_names:
        vars.append(get_var_maybe_avg(namespace, vn, training, polyak_decay))
    return vars


class WeightNormLinear(nn.Linear):
    """
    Implementation of "Weight Normalization: A Simple Reparameterization
    to Accelerate Training of Deep Neural Networks"
    :cite:`DBLP:journals/corr/SalimansK16`

    As a reparameterization method, weight normalization is same
    as BatchNormalization, but it doesn't depend on minibatch.

    NOTE: This is used nowhere in the code at this stage
          Vincent Nguyen 05/18/2018
    """

    def __init__(self, in_features, out_features, init_scale=1.0, polyak_decay=0.9995):
        super(WeightNormLinear, self).__init__(in_features, out_features, bias=True)

        self.V = self.weight
        self.g = Parameter(torch.Tensor(out_features))
        self.b = self.bias

        self.register_buffer("V_avg", torch.zeros(out_features, in_features))
        self.register_buffer("g_avg", torch.zeros(out_features))
        self.register_buffer("b_avg", torch.zeros(out_features))

        self.init_scale = init_scale
        self.polyak_decay = polyak_decay
        self.reset_parameters()

    def reset_parameters(self):
        return

    def forward(self, x, init=False):
        if init is True:
            # out_features * in_features
            self.V.data.copy_(
                torch.randn(self.V.data.size()).type_as(self.V.data) * 0.05
            )
            # norm is out_features * 1
            v_norm = self.V.data / self.V.data.norm(2, 1).expand_as(self.V.data)
            # batch_size * out_features
            x_init = F.linear(x, v_norm).data
            # out_features
            m_init, v_init = x_init.mean(0).squeeze(0), x_init.var(0).squeeze(0)
            # out_features
            scale_init = self.init_scale / torch.sqrt(v_init + 1e-10)
            self.g.data.copy_(scale_init)
            self.b.data.copy_(-m_init * scale_init)
            x_init = scale_init.view(1, -1).expand_as(x_init) * (
                x_init - m_init.view(1, -1).expand_as(x_init)
            )
            self.V_avg.copy_(self.V.data)
            self.g_avg.copy_(self.g.data)
            self.b_avg.copy_(self.b.data)
            return x_init
        else:
            v, g, b = get_vars_maybe_avg(
                self, ["V", "g", "b"], self.training, polyak_decay=self.polyak_decay
            )
            # batch_size * out_features
            x = F.linear(x, v)
            scalar = g / torch.norm(v, 2, 1).squeeze(1)
            x = scalar.view(1, -1).expand_as(x) * x + b.view(1, -1).expand_as(x)
            return x


class WeightNormConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        init_scale=1.0,
        polyak_decay=0.9995,
    ):
        super(WeightNormConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups
        )

        self.V = self.weight
        self.g = Parameter(torch.Tensor(out_channels))
        self.b = self.bias

        self.register_buffer("V_avg", torch.zeros(self.V.size()))
        self.register_buffer("g_avg", torch.zeros(out_channels))
        self.register_buffer("b_avg", torch.zeros(out_channels))

        self.init_scale = init_scale
        self.polyak_decay = polyak_decay
        self.reset_parameters()

    def reset_parameters(self):
        return

    def forward(self, x, init=False):
        if init is True:
            # out_channels, in_channels // groups, * kernel_size
            self.V.data.copy_(
                torch.randn(self.V.data.size()).type_as(self.V.data) * 0.05
            )
            v_norm = self.V.data / self.V.data.view(self.out_channels, -1).norm(
                2, 1
            ).view(self.out_channels, *([1] * (len(self.kernel_size) + 1))).expand_as(
                self.V.data
            )
            x_init = F.conv2d(
                x, v_norm, None, self.stride, self.padding, self.dilation, self.groups
            ).data
            t_x_init = x_init.transpose(0, 1).contiguous().view(self.out_channels, -1)
            m_init, v_init = t_x_init.mean(1).squeeze(1), t_x_init.var(1).squeeze(1)
            # out_features
            scale_init = self.init_scale / torch.sqrt(v_init + 1e-10)
            self.g.data.copy_(scale_init)
            self.b.data.copy_(-m_init * scale_init)
            scale_init_shape = scale_init.view(
                1, self.out_channels, *([1] * (len(x_init.size()) - 2))
            )
            m_init_shape = m_init.view(
                1, self.out_channels, *([1] * (len(x_init.size()) - 2))
            )
            x_init = scale_init_shape.expand_as(x_init) * (
                x_init - m_init_shape.expand_as(x_init)
            )
            self.V_avg.copy_(self.V.data)
            self.g_avg.copy_(self.g.data)
            self.b_avg.copy_(self.b.data)
            return x_init
        else:
            v, g, b = get_vars_maybe_avg(
                self, ["V", "g", "b"], self.training, polyak_decay=self.polyak_decay
            )

            scalar = torch.norm(v.view(self.out_channels, -1), 2, 1)
            if len(scalar.size()) == 2:
                scalar = g / scalar.squeeze(1)
            else:
                scalar = g / scalar

            w = (
                scalar.view(self.out_channels, *([1] * (len(v.size()) - 1))).expand_as(
                    v
                )
                * v
            )

            x = F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)
            return x


class CNNEncoder(nn.Module):
    """Encoder based on "Convolutional Sequence to Sequence Learning"
    :cite:`DBLP:journals/corr/GehringAGYD17`.
    """

    def __init__(self, num_layers, hidden_size, cnn_kernel_width, dropout, embeddings):
        super(CNNEncoder, self).__init__()

        self.embeddings = embeddings
        input_size = embeddings.embedding_size
        self.linear = nn.Linear(input_size, hidden_size)
        init.xavier_uniform_(self.linear.weight)
        self.cnn = StackedCNN(num_layers, hidden_size, cnn_kernel_width, dropout)

    def forward(self, input, src_len=None, hidden=None):
        """See :func:`EncoderBase.forward()`"""
        # batch x len x dim
        emb = self.embeddings(input)

        emb_reshape = emb.view(emb.size(0) * emb.size(1), -1)
        emb_remap = self.linear(emb_reshape)
        emb_remap = emb_remap.view(emb.size(0), emb.size(1), -1)
        emb_remap = shape_transform(emb_remap)
        out = self.cnn(emb_remap)

        return out.squeeze(3), emb_remap.squeeze(3), src_len

    def update_dropout(self, dropout, attention_dropout=None):
        self.cnn.dropout.p = dropout


class CNNDecoder(nn.Module):  # Inherit directly from nn.Module
    """Decoder based on "Convolutional Sequence to Sequence Learning"
    :cite:`DBLP:journals/corr/GehringAGYD17`.

    Consists of residual convolutional layers, with ConvMultiStepAttention.
    """

    def __init__(
        self,
        num_layers,
        hidden_size,
        attn_type,
        copy_attn,
        cnn_kernel_width,
        dropout,
        embeddings,
        copy_attn_type,
    ):
        super(CNNDecoder, self).__init__()

        self.cnn_kernel_width = cnn_kernel_width
        self.embeddings = embeddings

        # Decoder State
        self.state = {}

        input_size = self.embeddings.embedding_size
        self.linear = nn.Linear(input_size, hidden_size)
        init.xavier_uniform_(self.linear.weight)
        self.conv_layers = nn.ModuleList(
            [
                GatedConv(hidden_size, cnn_kernel_width, dropout, True)
                for i in range(num_layers)
            ]
        )
        self.attn_layers = nn.ModuleList(
            [ConvMultiStepAttention(hidden_size) for i in range(num_layers)]
        )

        # CNNDecoder has its own attention mechanism.
        # Set up a separate copy attention layer if needed.
        assert not copy_attn, "Copy mechanism not yet tested in conv2conv"
        if copy_attn:
            self.copy_attn = GlobalAttention(hidden_size, attn_type=copy_attn_type)
        else:
            self.copy_attn = None

    def init_state(self, _, enc_out, enc_hidden):
        """Init decoder state."""
        self.state["src"] = (enc_out + enc_hidden) * SCALE_WEIGHT
        self.state["previous_input"] = None

    def map_state(self, fn):
        self.state["src"] = fn(self.state["src"], 0)
        if self.state["previous_input"] is not None:
            self.state["previous_input"] = fn(self.state["previous_input"], 0)

    def detach_state(self):
        self.state["previous_input"] = self.state["previous_input"].detach()

    def forward(
        self, tgt, memory_bank, step=None, memory_lengths=None
    ):  # Updated signature
        """See :obj:`onmt.modules.RNNDecoderBase.forward()`"""
        if self.state["previous_input"] is not None:
            tgt = torch.cat([self.state["previous_input"], tgt], 1)

        dec_outs = []
        attns = {"std": []}
        if self.copy_attn is not None:
            attns["copy"] = []

        emb = self.embeddings(tgt)

        tgt_emb = emb
        enc_out_t = memory_bank.transpose(1, 2)  # Transpose memory_bank
        enc_out_c = self.state["src"]

        emb_reshape = tgt_emb.view(tgt_emb.size(0) * tgt_emb.size(1), -1)
        linear_out = self.linear(emb_reshape)
        x = linear_out.view(tgt_emb.size(0), tgt_emb.size(1), -1)
        x = shape_transform(x)

        pad = torch.zeros(x.size(0), x.size(1), self.cnn_kernel_width - 1, 1)

        pad = pad.type_as(x)
        base_target_emb = x

        for conv, attention in zip(self.conv_layers, self.attn_layers):
            new_target_input = torch.cat([pad, x], 2)
            out = conv(new_target_input)
            c, attn = attention(base_target_emb, out, enc_out_t, enc_out_c)
            x = (x + (c + out) * SCALE_WEIGHT) * SCALE_WEIGHT

        dec_outs = x.squeeze(3).transpose(1, 2)

        # Process the result and update the attentions.
        if self.state["previous_input"] is not None:
            dec_outs = dec_outs[:, self.state["previous_input"].size(1) :, :]
            attn = attn[:, self.state["previous_input"].size(1) :].squeeze()
            attn = torch.stack([attn])
        attns["std"] = attn
        if self.copy_attn is not None:
            attns["copy"] = attn

        # Update the state.
        self.state["previous_input"] = tgt
        # TODO change the way attns is returned dict => list or tuple (onnx)
        return dec_outs, attns

    def update_dropout(self, dropout, attention_dropout=None):
        for layer in self.conv_layers:
            layer.dropout.p = dropout

