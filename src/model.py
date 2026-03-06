import torch
import typing
import math

import os
import random
import numpy as np 

def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def move_to_device(data: typing.Any, device: torch.device) -> typing.Any:
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    if isinstance(data, dict):
        return {k: move_to_device(v, device=device) for k, v in data.items()}
    return data

@torch.inference_mode
def generate_batch(
    model : 'Transformer',
    src_c_nmr: typing.Dict[str, torch.Tensor],
    src_h_nmr: typing.Dict[str, torch.Tensor],
    tokenizer: 'SentencePieceProcessor',
    max_tokens: int = 200,
    device: torch.device = torch.device("cpu"),
) -> typing.List[str]:
    model.eval()

    batch_size = src_c_nmr["spectrum"].shape[0]
    idxs = torch.full(
        (batch_size, 1),
        fill_value=tokenizer.bos_id(),
        dtype=torch.int64,
        device=device,
    )
    system_tokens = {
        tokenizer.bos_id(),
        tokenizer.eos_id(),
        tokenizer.pad_id(),
        tokenizer.unk_id(),
    }

    src_c_nmr = move_to_device(src_c_nmr, device)
    src_h_nmr = move_to_device(src_h_nmr, device)

    while idxs.shape[-1] < max_tokens:
        logits = model(
            src_c_nmr=src_c_nmr,
            src_h_nmr=src_h_nmr,
            tgt=idxs,
        )
        new_idxs = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idxs = torch.cat([idxs, new_idxs], dim=-1)

        if torch.all(torch.any(idxs == tokenizer.eos_id(), dim=-1)):
            break

    idxs_list = []
    for seq in idxs.cpu().numpy().tolist():
        filtered = [tok for tok in seq if tok not in system_tokens]
        idxs_list.append(filtered)

    smiles_list = tokenizer.decode(idxs_list)
    return smiles_list

class FourierEmbedding(torch.nn.Module):
    def __init__(
        self,
        d_ff : int,
        output_dim : int
    ) -> None:
        super(FourierEmbedding, self).__init__()

        self.B = torch.nn.Parameter(torch.randn(1, d_ff//2), requires_grad=True)
        # self.B = torch.nn.Parameter(torch.randn(1, d_ff//2), requires_grad=False) 
        self.proj = torch.nn.Linear(d_ff, output_dim)

    def __postinit__(
        self
    ) -> None:
        self.B = self.B.to(next(iter(self.parameters())).device)

    def forward(
        self,
        x : torch.Tensor # [B, L]
    ) -> torch.Tensor: # [B, L, output_dim]
        x_proj = 2 * torch.pi * torch.einsum('bl,eo->blo', x, self.B) # [B, L, d_ff/2]
        fourier = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1) # [B, L, d_ff]
        return self.proj(fourier)

class SpectrumEmbedding(torch.nn.Module):
    def __init__(
        self,
        multiplicity_vocab_size : int,
        spectrum_d_ff : int,
        spectrum_hidden_dim : int,
        intensity_d_ff : int,
        intensity_hidden_dim : int,
        multiplicity_hidden_dim : int,
    ) -> None:
        super(SpectrumEmbedding, self).__init__()

        self.spectrum_embs : torch.nn.ModuleDict[str, torch.nn.Module] = torch.nn.ModuleDict(
            {
                "C_NMR" : FourierEmbedding(
                    d_ff=spectrum_d_ff,
                    output_dim=spectrum_hidden_dim
                ),
                "H_NMR" : FourierEmbedding(
                    d_ff=spectrum_d_ff,
                    output_dim=spectrum_hidden_dim
                ),
            }
        )

        self.intensity_embs : torch.nn.ModuleDict[str, torch.nn.Module] = torch.nn.ModuleDict(
            {
                "C_NMR" : FourierEmbedding(
                    d_ff=intensity_d_ff,
                    output_dim=intensity_hidden_dim
                ),
                "H_NMR" : FourierEmbedding(
                    d_ff=intensity_d_ff,
                    output_dim=intensity_hidden_dim
                ),
            }
        )

        self.multiplicity_embs : torch.nn.ModuleDict[str, torch.nn.Module] = torch.nn.ModuleDict(
            {
                "C_NMR" : torch.nn.Embedding(multiplicity_vocab_size, multiplicity_hidden_dim),
                "H_NMR" : torch.nn.Embedding(multiplicity_vocab_size, multiplicity_hidden_dim),
            }
        )

    def forward(
        self,
        data : typing.Dict[str, torch.Tensor],
        spectrum_type : typing.Literal["C_NMR", "H_NMR"]
    ) -> torch.Tensor:

        if spectrum_type not in {"C_NMR", "H_NMR"}:
            raise ValueError(f"Expected spectrum_type C_NMR or H_NMR, got {spectrum_type}")
        
        spectrum_emb = self.spectrum_embs[spectrum_type](data['spectrum'])
        intensity_emb = self.intensity_embs[spectrum_type](data['intensity'])
        multiplicity_emb = self.multiplicity_embs[spectrum_type](data['multiplicity'])
        
        return torch.cat([spectrum_emb, intensity_emb, multiplicity_emb], dim=-1)

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.W_o = torch.nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            if mask.dtype == torch.bool:
                # mask == True means keep, False means mask out
                mask_to_apply = ~mask
            else:
                # mask == 0 means mask out
                mask_to_apply = (mask == 0)
            attn_scores = attn_scores.masked_fill(
                mask_to_apply,
                torch.finfo(attn_scores.dtype).min,
            )
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = torch.nn.Linear(d_model, d_ff)
        self.fc2 = torch.nn.Linear(d_ff, d_model)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(
        self, 
        x : torch.Tensor,
        mask : torch.Tensor
    ) -> torch.Tensor:
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.carbon_cross_attn = MultiHeadAttention(d_model, num_heads)
        self.hydrogen_cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.norm3 = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(
        self, 
        x : torch.Tensor, 
        enc_carbon_output : torch.Tensor, 
        enc_hydrogen_output : torch.Tensor, 
        src_carbon_mask : torch.Tensor, 
        src_hydrogen_mask : torch.Tensor, 
        tgt_mask : torch.Tensor
    ) -> torch.Tensor:

        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        carbon_attn_output = self.carbon_cross_attn(
            Q=x, 
            K=enc_carbon_output, 
            V=enc_carbon_output, 
            mask=src_carbon_mask
        )
        
        hydrogen_attn_output = self.hydrogen_cross_attn(
            Q=x, 
            K=enc_hydrogen_output, 
            V=enc_hydrogen_output, 
            mask=src_hydrogen_mask
        )

        carbon_attn_output = carbon_attn_output.masked_fill(~src_carbon_mask.any(dim=-1), 0.)
        hydrogen_attn_output = hydrogen_attn_output.masked_fill(~src_hydrogen_mask.any(dim=-1), 0.)
        
        x = self.norm2(x + self.dropout(carbon_attn_output) + self.dropout(hydrogen_attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(torch.nn.Module):
    def __init__(
        self, 
        spectrum_d_ff : int,
        spectrum_hidden_dim : int,
        intensity_d_ff : int,
        intensity_hidden_dim : int,
        multiplicity_vocab_size : int,
        multiplicity_hidden_dim : int,
        tgt_vocab_size : int, 
        d_model : int, 
        num_heads : int, 
        num_layers : int, 
        d_ff : int, 
        max_seq_length : int, 
        dropout : float,
        smiles_pad_token : int
    ) -> None:
        super(Transformer, self).__init__()
        
        self.spectrum_embedding = SpectrumEmbedding(
            multiplicity_vocab_size=multiplicity_vocab_size,
            multiplicity_hidden_dim=multiplicity_hidden_dim,
            spectrum_d_ff=spectrum_d_ff,
            spectrum_hidden_dim=spectrum_hidden_dim,
            intensity_d_ff=intensity_d_ff,
            intensity_hidden_dim=intensity_hidden_dim,
        )
        
        self.decoder_embedding = torch.nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_carbon_layers = torch.nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.encoder_hydrogen_layers = torch.nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.decoder_layers = torch.nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = torch.nn.Linear(d_model, tgt_vocab_size)
        self.dropout = torch.nn.Dropout(dropout)

        self.smiles_pad_token = smiles_pad_token
        self.register_buffer(
            "cached_nopeak_mask",
            torch.tril(torch.ones(1, max_seq_length, max_seq_length, dtype=torch.bool)),
            persistent=False,
        )

    def generate_mask(
        self, 
        src_carbon_spectrum : torch.Tensor,
        src_hydrogen_spectrum : torch.Tensor,
        tgt : torch.Tensor
    ):
        src_carbon_mask = (src_carbon_spectrum >= 0).unsqueeze(1).unsqueeze(2)
        src_hydrogen_mask = (src_hydrogen_spectrum >= 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != self.smiles_pad_token).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = self.cached_nopeak_mask[:, :seq_length, :seq_length]
        tgt_mask = tgt_mask & nopeak_mask
        return src_carbon_mask, src_hydrogen_mask, tgt_mask

    def forward(
        self, 
        src_c_nmr : typing.Dict[str, torch.Tensor], 
        src_h_nmr : typing.Dict[str, torch.Tensor],
        tgt : torch.Tensor
    ):
        src_carbon_mask, src_hydrogen_mask, tgt_mask = self.generate_mask(
            src_carbon_spectrum=src_c_nmr['spectrum'],
            src_hydrogen_spectrum=src_h_nmr['spectrum'],
            tgt=tgt
        )
        
        src_c_nmr_embedded = self.dropout(self.positional_encoding(self.spectrum_embedding(src_c_nmr, spectrum_type='C_NMR')))
        src_h_nmr_embedded = self.dropout(self.positional_encoding(self.spectrum_embedding(src_h_nmr, spectrum_type='H_NMR')))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_carbon_output = src_c_nmr_embedded
        for enc_layer in self.encoder_carbon_layers:
            enc_carbon_output = enc_layer(enc_carbon_output, src_carbon_mask)

        enc_hydrogen_output = src_h_nmr_embedded
        for enc_layer in self.encoder_hydrogen_layers:
            enc_hydrogen_output = enc_layer(enc_hydrogen_output, src_hydrogen_mask)
        
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(
                x=dec_output, 
                enc_carbon_output=enc_carbon_output,
                enc_hydrogen_output=enc_hydrogen_output,
                src_carbon_mask=src_carbon_mask, 
                src_hydrogen_mask=src_hydrogen_mask, 
                tgt_mask=tgt_mask
            )

        output = self.fc(dec_output)
        return output
