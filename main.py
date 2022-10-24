import torch
from torch import nn


def attentionFunction(in_size, out_size, additiveBias=True):
    return nn.Linear(in_size, out_size, additiveBias)


class SelfAttention(nn.Module):
    """
            ........       ,=============,
    V--->[Linear]'---->,-------------,||
        ........     |   Scaled    |||
    K--->[Linear]'---->| Dot-Product |---->[Concat]---->[Linear]---> MHA
        ........     |  Attention  |_|
    Q--->[Linear]'---->|_____________|

        ,--------,
    Q-->|        |                                     ,--------,
        | MatMul |->[Scale]->[Mask(opt.)]->[SoftMax]-->|        |    Scaled
    K-->|________|                                     | MatMul |--> Dot-
                                                       |        |    Product
    V------------------------------------------------->|________|    Attention
    """
    def __init__(self, embed_size, head_count):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.head_count = head_count
        self.head_dim = embed_size // head_count
        assert (self.head_dim * head_count == embed_size), \
            "Embed size need to be divisible by Head count."

        self.keys = attentionFunction(self.head_dim, self.head_dim, False)
        self.values = attentionFunction(self.head_dim, self.head_dim, False)
        self.queries = attentionFunction(self.head_dim, self.head_dim, False)
        self.queries = attentionFunction(self.head_dim, self.head_dim, False)
        self.full_connected_out = attentionFunction(
            self.head_count*self.head_dim, self.embed_size)

    def forward(self, keys, values, query, mask):
        #  number of training examples
        N = query.shape[0]
        #  split embedding into heads pieces
        keys, keys_count = self.split_into_heads(keys, N)
        values, values_count = self.split_into_heads(values, N)
        queries, queries_count = self.split_into_heads(query, N)

        keys = self.keys(keys)
        values = self.values(values)
        queries = self.queries(queries)

        # Attention(Q, K, V) = softmax(QK / sqrt(d))V

        # queries shape: (N, query_len, head_count, head_dim) :: n,q,h,d
        # keys shape:    (N, key_len, head_count, head_dim)   :: n,k,h,d
        # energy shape:  (N, heads, query_len, key_len)       :: n,h,q,k
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        softmax_result = torch.softmax(energy / (self.embed_size ** 0.5),
                                       dim=3)

        # softmax_result shape:    (N, heads, query_len, key_len)       :: nhqk
        # values shape:            (N, value_len, head_count, head_dim) :: nvhd
        # attention shape:         (N, query_len, head_count, head_dim) :: nqhd
        attention = torch.einsum("nhqk,nvhd->nqhd", [softmax_result, values])
        # flattening last 2 dimensions
        attention = attention.reshape(N,
                                      queries_count,
                                      self.head_count*self.head_dim)

        out = self.full_connected_out(attention)
        return out

    def split_into_heads(self, obj, N):
        obj_count = obj.shape[1]
        obj = obj.reshape(N, obj_count, self.head_count, self.head_dim)
        return obj, obj_count


class TransformerBlock(nn.Module):
    """
    ,---------------,    ,----------,                      --,
    |               |,   |          |,                       |
    -=-=->[ MHA ]--[A&N]--=>[ FF ]--[A&N]--.                  ~Nx
        \\__/'                              |                  |
                            ._____________|                --'
    ,--------------,        |  |
    """
    def __init__(self, embed_size, head_count, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, head_count)
        self.norm_first = nn.LayerNorm(embed_size)
        self.norm_second = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

        forward_expansion_for_size = forward_expansion * embed_size
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion_for_size),
            nn.ReLU(),
            nn.Linear(forward_expansion_for_size, embed_size),
        )

    def forward(self, key, value, query, mask):
        attention = self.attention(key, value, query, mask)
        x = self.dropout(self.norm_first(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm_second(forward + x))
        return out


class Encoder(nn.Module):
    """
                           ,--------------, --,
                       PE  | Transformer  |   |
 Inputs-->[Input    ]-(+)--=     Block    |   ~Nx
          [Embedding]      |              |   |
                           |______________| --'
    """
    def __init__(self, src_vocab_size, embed_size, num_layers, head_count,
                 device, dropout, forward_expansion, max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [TransformerBlock(
                embed_size,
                head_count,
                dropout,
                forward_expansion,
            ) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len). \
            expand(N, seq_len). \
            to(self.device)
        out = self.dropout(
            self.word_embedding(x) + self.position_embedding(positions),
        )

        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out


class DecoderBlock(nn.Module):
    """
    ,--------------,        |, |,
    |              |      .--------------.
    |    [Masked]  |,     |  Transformer |    --,
    --'-=->[ MHA  ]-[A&N]-|  Block       |      ~Nx
         \\__/'           |______________|    --'
    """
    def __init__(self, embed_size, head_count,
                 device, dropout, forward_expansion):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, head_count)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size,
            head_count,
            dropout,
            forward_expansion,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key, value, src_mask, target_mask):
        attention = self.attention(x, x, x, target_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(key, value, query, src_mask)
        return out


class Decoder(nn.Module):
    """
                                .--------------.
    Outputs                PE   |  Decoder     |
    (shifted)->[Output   ]-(+)--|  Block       |
    (right  )  [Embedding]      |______________|
    """
    def __init__(self, target_vocab_size, embed_size, num_layers, head_count,
                 device, dropout, forward_expansion, max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(target_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [DecoderBlock(
                embed_size,
                head_count,
                device,
                dropout,
                forward_expansion,
            ) for _ in range(num_layers)]
        )

        self.full_connected_out = nn.Linear(embed_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out, src_mask, target_mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len). \
            expand(N, seq_len). \
            to(self.device)
        x = self.dropout(
            self.word_embedding(x) + self.position_embedding(positions),
        )

        for layer in self.layers:
            x = layer(x, encoder_out, encoder_out, src_mask, target_mask)
        out = self.full_connected_out(x)
        return out


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, target_vocab_size,
                 src_pad_idx, target_pad_idx,
                 embed_size=256, num_layers=6, head_count=8,
                 device="cpu", dropout=0, forward_expansion=4, max_length=100):
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.target_pad_idx = target_pad_idx
        self.device = device

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            head_count,
            device,
            dropout,
            forward_expansion,
            max_length,
        )

        self.decoder = Decoder(
            target_vocab_size,
            embed_size,
            num_layers,
            head_count,
            device,
            dropout,
            forward_expansion,
            max_length,
        )

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_target_mask(self, target):
        N, target_length = target.shape
        target_mask = torch.tril(torch.ones((target_length, target_length))). \
            expand(N, 1, target_length, target_length)
        return target_mask.to(self.device)

    def forward(self, src, target):
        src_mask = self.make_src_mask(src)
        target_mask = self.make_target_mask(target)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(target, enc_src, src_mask, target_mask)
        return out


if __name__ == "__main__":
    device = "cpu"
    #try:
    #    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    #except Exception as e:
    #    print(e)
    #    print("no CUDA, so CPU")

    x = torch.tensor([[1,5,6,4,3,9,5,2,0], [1,8,7,3,4,5,6,7,2]]).to(device)
    target =  torch.tensor([[1,7,4,3,5,9,2,0], [1,5,6,2,4,7,6,2]]).to(device)

    src_pad_idx, target_pad_idx = 0, 0
    src_vocab_size, target_vocab_size = 10, 10

    model = Transformer(
        src_vocab_size,
        target_vocab_size,
        src_pad_idx,
        target_pad_idx
    ).to(device)
    out = model(x, target[:, :-1])
    print(out.shape)
