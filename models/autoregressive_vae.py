import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# model
class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if x.dim() == 3:
            B, T, _ = x.shape
        elif x.dim() == 2:
            B, T = x.shape
        if T <= self.pe.size(0):
            pe = self.pe[:T]  
        else:
            device = x.device
            H = self.hidden_size
            position = torch.arange(T, dtype=torch.float, device=device).unsqueeze(1)  
            div_term = torch.exp(torch.arange(0, H, 2, device=device).float() * (-math.log(10000.0)/H))
            pe = torch.zeros(T, H, device=device)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, correct_head_layout=False):
        super().__init__()
        self.d = hidden_size // num_heads
        self.num_heads = num_heads
        self.correct_head_layout = correct_head_layout
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Dropout(p=0.1)
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, q, k, v, pad_mask=None, causal=False):   # [B, T, H]
        B, T_q, H = q.shape
        _, T_v, _ = v.shape
        Q = self.W_q(q)     # [B, T, num_heads * H]
        K = self.W_k(k)
        Vx = self.W_v(v)
        if self.correct_head_layout:
            Q = Q.view(B, T_q, self.num_heads, self.d).transpose(1, 2)
            K = K.view(B, T_v, self.num_heads, self.d).transpose(1, 2)
            V = Vx.view(B, T_v, self.num_heads, self.d).transpose(1, 2)
        else:
            # Preserve released checkpoint behavior. This legacy reshape mixes
            # token and head axes and therefore depends on padded sequence width.
            Q = Q.view(B, self.num_heads, T_q, self.d)
            K = K.view(B, self.num_heads, T_v, self.d)
            V = Vx.view(B, self.num_heads, T_v, self.d)

        attn_logits = torch.einsum('baih,bajh->baij', Q, K)    # [B, A, T, H] @ [B, A, H, T] = [B, A, T, T]

        if pad_mask is not None:
            key_mask = pad_mask[:, None, None, :]  # [B,1,1,T_k]
            attn_logits = attn_logits.masked_fill(~key_mask, float('-inf'))

        attn = F.softmax(attn_logits / math.sqrt(self.d), dim=-1)

        if pad_mask is not None:
            query_mask = pad_mask[:, None, :, None]  # [B,1,T_q,1]
            attn = attn * query_mask.float()

        h = torch.einsum('baij,bajh->baih',attn, V)  # [B, A, T, H]
        if self.correct_head_layout:
            h = h.transpose(1, 2).contiguous().view(B, T_q, H)
        else:
            h = h.view(B, T_q, H)  # legacy [B, A, T, D] reinterpretation

        # soft XSA
        # Vn = F.normalize(Vx, dim=-1)
        # h = h - 0.8 * (h * Vn).sum(dim=-1, keepdim=True) * Vn
        
        h = q + self.W_o(h)     # [B, T, H]
        h = h + self.ff(self.norm2(h))
        h = self.norm1(h)
        return h

class MultiSlotPooling(nn.Module):
    def __init__(self, hidden_size, num_slots):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(num_slots, hidden_size))
        nn.init.uniform_(self.queries, a=-1.0, b=1.0)        
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, h, mask):
        # h: [B, T, D]
        # mask: [B, T]
        k = self.W_k(h)
        v = self.W_v(h)
        attn = torch.einsum("kd,btd->bkt", self.queries, k)
        attn = attn.masked_fill(~mask[:, None, :], -1e9)
        attn = F.softmax(attn, dim=-1) / math.sqrt(h.size(-1))
        slots = torch.einsum("bkt,btd->bkd", attn, v)
        return slots
    

class VaeTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        latent_size,
        max_len,
        attn_heads=8,
        num_slots=8,
        encoder_layers=1,
        decoder_layers=1,
        padding_invariant_encoder=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_slots = num_slots
        # False preserves the behavior of released/legacy checkpoints whose
        # configurations predate this flag. New training code explicitly sets
        # this to True so right-padding cannot affect the encoder latent.
        self.padding_invariant_encoder = padding_invariant_encoder
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoder = PositionalEmbedding(max_len, hidden_size)
        self.conv_pos = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size)
        
        # Encoder
        self.pos_block = MultiHeadAttention(
            hidden_size, attn_heads, correct_head_layout=padding_invariant_encoder
        )
        self.encoder_blocks = nn.ModuleList(
            [
                MultiHeadAttention(
                    hidden_size,
                    attn_heads,
                    correct_head_layout=padding_invariant_encoder,
                )
                for _ in range(encoder_layers)
            ]
        )
        self.pool = MultiSlotPooling(hidden_size, num_slots=num_slots)
        # These checkpointed modules/parameters are part of the released training
        # architecture.  They are retained for strict checkpoint compatibility even
        # though the final encoder path below does not invoke the optional slot mixer.
        self.slots_mix = MultiHeadAttention(
            hidden_size,
            num_heads=num_slots,
            correct_head_layout=padding_invariant_encoder,
        )
        self.slot_gamma = nn.Parameter(torch.ones(1, num_slots, hidden_size))
        self.slot_beta = nn.Parameter(torch.zeros(1, num_slots, hidden_size))

        # VAE heads
        self.slot_mu = nn.Linear(hidden_size, latent_size)
        self.slot_logvar = nn.Linear(hidden_size, latent_size)  
        self.slot_compress_mu = nn.Linear(hidden_size, latent_size // num_slots)
        self.slot_compress_logvar = nn.Linear(hidden_size, latent_size // num_slots)

        self.fc_mu = nn.Linear(num_slots * hidden_size, latent_size)
        self.fc_logvar = nn.Linear(num_slots * hidden_size, latent_size)
        
        # pooling in latent space with uncertanty
        self.latent_query = nn.Parameter(torch.randn(1, 1, latent_size))
        self.latent_key = nn.Linear(latent_size, latent_size)

        # Decoder
        self.max_len = max_len
        self.z_to_slot = nn.Linear(hidden_size // num_slots, hidden_size)
        #self.slot_pos_decoder = nn.Parameter(torch.randn(1, num_slots, hidden_size))
        self.decoder_embed = nn.Embedding(vocab_size, hidden_size)

        self.decoder_pos = PositionalEmbedding(max_len, hidden_size)

        self.z_to_memory = nn.Linear(latent_size, hidden_size)
        self.slots_to_memory = nn.Linear(latent_size // num_slots, hidden_size)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=attn_heads,
            dim_feedforward=2 * hidden_size,
            batch_first=True
        )

        self.decoder_transformer = nn.TransformerDecoder(
            self.decoder_layer,
            num_layers=decoder_layers
        )
        # Output head
        self.fc_output = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def causal_mask(self, T, device):
        return torch.triu(
            torch.ones(T, T, device=device),
            diagonal=1
        ).bool()
    
    def encode(self, x, mode=None):  
        B, _ = x.shape
        mask = (x != 0)
        h = self.embedding(x)    # [B, T, H]
        if self.padding_invariant_encoder:
            h = h.masked_fill(~mask[:, :, None], 0.0)
        fourier_pos_encoding = self.pos_encoder(x)
        conv_pos_encoding = self.conv_pos(h.transpose(1, 2)).transpose(1, 2)
        h = h + fourier_pos_encoding + conv_pos_encoding
        
        for block in self.encoder_blocks:
            h = block(h, h, h, pad_mask=mask)
            
        h = h.masked_fill(~mask[:, :, None], 0.0)

        slots = self.pool(h, mask)
        slots = F.layer_norm(slots, slots.shape[-1:])
        #slots = slots * self.slot_gamma + self.slot_beta

        slots_mu = self.slot_mu(slots)
        slots_logvar = self.slot_logvar(slots)

        B, K, Z = slots_mu.shape

        # Variance aggregation and exponentials are numerically sensitive under
        # BF16. Keep this small latent-space calculation in FP32 and constrain
        # log-variance to a useful, finite range.
        with torch.autocast(device_type=h.device.type, enabled=False):
            slots_mu_fp32 = slots_mu.float()
            slots_logvar_fp32 = slots_logvar.float().clamp(min=-12.0, max=6.0)
            q = F.normalize(self.latent_query.float().expand(B, 1, Z), dim=-1)
            k = F.normalize(self.latent_key(slots_mu_fp32), dim=-1)
            logits = torch.einsum("bqz,bkz->bqk", q, k)
            confidence = -torch.logsumexp(slots_logvar_fp32, dim=-1).unsqueeze(1)
            attn = torch.softmax((logits + 0.5 * confidence) / 0.5, dim=-1)
            mu = torch.einsum("bqk,bkz->bqz", attn, slots_mu_fp32).squeeze(1)
            var = torch.exp(slots_logvar_fp32)
            var_agg = torch.einsum("bqk,bkz->bqz", attn, var).squeeze(1)
            logvar = torch.log(var_agg.clamp_min(1e-8)).clamp(min=-12.0, max=6.0)
        
        if mode == "test":
            return mu, logvar, slots
        else:
            return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar.float().clamp(min=-12.0, max=6.0))
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z, x_in=None, max_len=None, start_id=1, eos_id=2):
        B = z.size(0)
        device = z.device

        memory = self.z_to_memory(z).unsqueeze(1)

        if x_in is not None:
            x_emb = self.decoder_embed(x_in)
            x_emb = x_emb + self.decoder_pos(x_emb)

            T = x_emb.size(1)
            tgt_mask = self.causal_mask(T, device)

            h = self.decoder_transformer(
                tgt=x_emb,
                memory=memory,
                tgt_mask=tgt_mask
            )

            logits = self.fc_output(h)   # ✔️ [B, T, V]
            return logits

        else:
            tokens = torch.full((B, 1), start_id, dtype=torch.long, device=device)
            finished = torch.zeros(B, dtype=torch.bool, device=device)
            max_len = self.max_len if max_len is None else int(max_len)

            for _ in range(max_len):
                x_emb = self.decoder_embed(tokens)
                x_emb = x_emb + self.decoder_pos(x_emb)

                T = tokens.size(1)
                tgt_mask = self.causal_mask(T, device)

                h = self.decoder_transformer(
                    tgt=x_emb,
                    memory=memory,
                    tgt_mask=tgt_mask
                )

                logits_step = self.fc_output(h[:, -1])  # [B, V]

                next_token = torch.argmax(logits_step, dim=-1, keepdim=True)

                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_token, eos_id),
                    next_token
                )

                tokens = torch.cat([tokens, next_token], dim=1)

                finished |= (next_token.squeeze(1) == eos_id)

                if finished.all():
                    break

            return tokens
            
    def forward(self, x, mode='eval'):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        if mode == 'train':
            x_in = x[:, :-1]

            logits = self.decode(z, x_in=x_in)

            return logits, mu, logvar
    

        if mode == "eval":
            tokens = self.decode(z, x_in=None)

            return tokens, mu, logvar 
        
        if mode == "test":
            mu, logvar, slots = self.encode(x, mode="test")
            z = self.reparameterize(mu, logvar)

            tokens = self.decode(z, x_in=None)

            return tokens, mu, logvar, slots


def vae_loss(logits, targets, mu, logvar, beta=0.01, pad_id=0):
    B, T, V = logits.shape

    logits = logits.reshape(-1, V)
    targets = targets.reshape(-1)

    rec_loss = F.cross_entropy(
        logits,
        targets,
        ignore_index=pad_id
    )

    # Evaluate KL in FP32 even when reconstruction runs under autocast.
    mu_fp32 = mu.float()
    logvar_fp32 = logvar.float().clamp(min=-12.0, max=6.0)
    kl = 0.5 * torch.mean(
        mu_fp32.square() + torch.exp(logvar_fp32) - 1.0 - logvar_fp32
    )

    return rec_loss + beta * kl, rec_loss, kl
