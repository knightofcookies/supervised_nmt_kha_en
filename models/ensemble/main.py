from collections import Counter
import math
import warnings
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.nn.init as init
from torch.nn.utils import skip_init
import sentencepiece as spm  # For tokenization
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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


class PositionalEncoding(nn.Module):
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
        super(PositionalEncoding, self).__init__()
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
                f"Sequence is {emb.size(1) + step} but PositionalEncoding is"
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
        position_encoding (bool): see :class:`~onmt.modules.PositionalEncoding`
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
            pe = PositionalEncoding(self.embedding_size, position_encoding_type)
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


# 1. Vocabulary Creation (using SentencePiece-tokenized files):


def build_vocab(filepath, tokenizer):
    counter = Counter()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            tokens = tokenizer.EncodeAsPieces(line.strip())  # Tokenize each line
            counter.update(tokens)

    sorted_vocab = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    vocab = {"<unk>": 0, "<blank>": 1, "<sos>": 2, "<eos>": 3}  # Add special tokens
    for i, (token, _) in enumerate(sorted_vocab):
        vocab[token] = i + 4  # Start after special tokens
    return vocab


en_model_prefix = "en_multi30k_bpe"
de_model_prefix = "de_multi30k_bpe"

en_sp = spm.SentencePieceProcessor()
de_sp = spm.SentencePieceProcessor()

en_sp.Load(f"{en_model_prefix}.model")
de_sp.Load(f"{de_model_prefix}.model")


# 2. Create Embeddings instances:

word_vec_size = 512
en_vocab_size = en_sp.GetPieceSize()
de_vocab_size = de_sp.GetPieceSize()
en_word_padding_idx = en_sp.pad_id()
de_word_padding_idx = de_sp.pad_id()
dropout = 0.1
position_encoding = True


en_embeddings = Embeddings(
    word_vec_size,
    en_vocab_size,
    en_word_padding_idx,
    position_encoding=position_encoding,
    dropout=dropout,
)


de_embeddings = Embeddings(
    word_vec_size,
    de_vocab_size,
    de_word_padding_idx,
    position_encoding=position_encoding,
    dropout=dropout,
)


for embedding in en_embeddings.emb_luts:
    init.xavier_uniform_(embedding.weight)
for embedding in de_embeddings.emb_luts:
    init.xavier_uniform_(embedding.weight)

# 3. Numericalize input and get embeddings:


def numericalize(text, tokenizer):
    ids = tokenizer.EncodeAsIds(text)
    return torch.tensor(ids)


en_encoder = CNNEncoder(
    num_layers=3,
    hidden_size=512,
    cnn_kernel_width=3,
    dropout=0.1,
    embeddings=en_embeddings,
)
de_encoder = CNNEncoder(
    num_layers=3,
    hidden_size=512,
    cnn_kernel_width=3,
    dropout=0.1,
    embeddings=de_embeddings,
)

en_decoder = CNNDecoder(
    num_layers=3,
    hidden_size=512,
    attn_type="general",
    copy_attn=False,
    cnn_kernel_width=3,
    dropout=0.1,
    embeddings=en_embeddings,
    copy_attn_type="general",
)  # Example values. Adjust as needed.
de_decoder = CNNDecoder(
    num_layers=3,
    hidden_size=512,
    attn_type="general",
    copy_attn=False,
    cnn_kernel_width=3,
    dropout=0.1,
    embeddings=de_embeddings,
    copy_attn_type="general",
)


# Output Layer
output_layer_de = nn.Linear(512, de_vocab_size)

# Loss Function
criterion = nn.CrossEntropyLoss(ignore_index=de_word_padding_idx)


# Training Loop
def train_en_to_de(
    en_encoder,
    de_decoder,
    output_layer_de,
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

            en_encoded, en_remap, _ = en_encoder(en_input, en_lengths)

            # Decoder training (English to German) using Teacher Forcing
            de_decoder.init_state(
                None, en_encoded, en_remap
            )  # Use English encoding to initialize German decoder

            de_target_input = de_input[:, :-1].to(
                device
            )  # Shift target for teacher forcing
            de_target_output = de_input[:, 1:].to(device)

            de_decoded_output, _ = de_decoder(
                de_target_input, en_encoded, memory_lengths=en_lengths
            )  # Use English encoded output as memory bank
            output = output_layer_de(de_decoded_output)

            loss = criterion(
                output.contiguous().view(-1, de_vocab_size),
                de_target_output.contiguous().view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        end_time = time.time()
        epoch_time = end_time - start_time

        total_loss = total_loss / len(en_train)

        print(
            f"Epoch: {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Time: {epoch_time:.2f}s"
        )


optimizer_en_de = optim.Adam(
    list(en_encoder.parameters())
    + list(de_decoder.parameters())
    + list(output_layer_de.parameters()),
    lr=0.0001,
)

en_encoder.to(device)
de_decoder.to(device)
output_layer_de.to(device)
criterion.to(device)

# train_en_to_de(
#     en_encoder,
#     de_decoder,
#     output_layer_de,
#     criterion,
#     optimizer_en_de,
#     en_sp,
#     de_sp,
#     num_epochs=30,
#     batch_size=128,
# )


def evaluate_en_to_de(
    en_encoder,
    de_decoder,
    output_layer_de,
    en_sp,
    de_sp,
    en_input,
):
    """Evaluate English to German translation."""
    en_encoder.eval()
    de_decoder.eval()
    with torch.no_grad():
        en_numericalized = numericalize(en_input, en_sp).unsqueeze(0).to(device)
        en_length = torch.tensor([en_numericalized.shape[1]]).to(device)
        en_input = nn.utils.rnn.pad_sequence(
            en_numericalized, padding_value=en_word_padding_idx, batch_first=True
        ).to(device)

        en_encoded, en_remap, en_lengths_output = en_encoder(en_input, en_length)

        de_decoder.init_state(None, en_encoded, en_remap)

        de_decoded_words = []
        de_prev_word = torch.tensor([[de_sp.bos_id()]]).to(device)

        for _ in range(en_length + 5):  # Max output length
            de_decoder_output, _ = de_decoder(
                de_prev_word, en_encoded, memory_lengths=en_lengths_output
            )

            output = output_layer_de(de_decoder_output)
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

    en_encoder.train()
    de_decoder.train()
    return "".join(de_decoded_words).replace("▁", " ")


en_input_sentence = "A little girl climbing into a wooden playhouse."
translated_sentence_de = evaluate_en_to_de(
    en_encoder,
    de_decoder,
    output_layer_de,
    en_sp,
    de_sp,
    en_input_sentence,
)
print(f"Translated Sentence (German): {translated_sentence_de}")
