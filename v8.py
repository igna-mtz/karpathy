import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):

    def __init__(self, n_emb, head_size, block_size, dropout):
        super().__init__()

        self.head_size = head_size

        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5

        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)

        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, n_emb, num_heads, head_size, block_size, dropout):
        super().__init__()

        self.heads = nn.ModuleList([
            Head(
                n_emb=n_emb,
                head_size=head_size,
                block_size=block_size,
                dropout=dropout
            )
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(
            num_heads * head_size,
            n_emb
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat(
            [h(x) for h in self.heads],
            dim=-1
        )

        return self.dropout(self.proj(out))


class FeedForward(nn.Module):

    def __init__(self, n_emb, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_emb, n_head, block_size, dropout):
        super().__init__()

        head_size = n_emb // n_head

        self.sa = MultiHeadAttention(
            n_emb=n_emb,
            num_heads=n_head,
            head_size=head_size,
            block_size=block_size,
            dropout=dropout
        )

        self.ffw = FeedForward(
            n_emb=n_emb,
            dropout=dropout
        )

        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        block_size,
        n_emb,
        n_head,
        n_layer,
        dropout
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(
            vocab_size,
            n_emb
        )

        self.position_embedding_table = nn.Embedding(
            block_size,
            n_emb
        )

        self.blocks = nn.Sequential(*[
            Block(
                n_emb=n_emb,
                n_head=n_head,
                block_size=block_size,
                dropout=dropout
            )
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)

        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            B, T, C = logits.shape

            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)

            loss = F.cross_entropy(
                logits_flat,
                targets_flat
            )

        return logits, loss

    def generate(self, idx, max_new_tokens):

        was_training = self.training
        self.eval()

        with torch.no_grad():

            for _ in range(max_new_tokens):

                idx_cond = idx[:, -self.block_size:]

                logits, _ = self(idx_cond)

                logits = logits[:, -1, :]

                probs = F.softmax(logits, dim=-1)

                idx_next = torch.multinomial(
                    probs,
                    num_samples=1
                )

                idx = torch.cat(
                    (idx, idx_next),
                    dim=1
                )

        if was_training:
            self.train()

        return idx