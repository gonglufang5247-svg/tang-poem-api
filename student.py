"""The six functions YOU write.

Each docstring says what the function does and exactly what goes in and out.
Run `python test_student.py` after you write them. Do not go on until all six pass.

These functions are the only thing we take from your folder and run with our
own copy of `model.py`, so they must do exactly what the docstring says.
"""
import torch

def softmax(logits):
    """Turn scores into probabilities.

    In:  logits — any shape of float scores, e.g. (B, T, V) or (V,).
    Out: same shape; each slice along the LAST dimension is non-negative and
         sums to 1. Must stay finite for scores like 1000.
    """
    max_val, _ = torch.max(logits, dim=-1, keepdim=True)
    exp_logits = torch.exp(logits - max_val)
    return exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)


def loss(logits, targets):
    """Cross-entropy: how surprised the model was by the real next character.

    In:  logits  — (B, T, V) scores from the model.
         targets — (B, T) integer character ids (the real next characters).
    Out: a scalar — the MEAN over all B*T positions, in natural log (nats),
         like torch.nn.functional.cross_entropy.
    """
    B, T, V = logits.shape
    return torch.nn.functional.cross_entropy(logits.contiguous().view(B * T, V), targets.contiguous().view(B * T))


def add_positions(x, pos):
    """Add position information to token embeddings.

    In:  x   — (B, T, C) token embeddings.
         pos — (max_T, C) position embedding matrix (model.pos.weight).
    Out: (B, T, C) — x with position t receiving row pos[t].
    """
    T = x.shape[1]
    return x + pos[:T, :]


def causal_mask(block_size):
    """The mask that stops each character from seeing the characters after it.

    In:  block_size — an integer T.
    Out: a (T, T) boolean tensor, ON THE CPU, where entry [i, j] is True when
         position i is ALLOWED to look at position j (j <= i) and False
         otherwise. model.py moves it to the right device for you.
    """
    return torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))


def attention(q, k, v, mask):
    """Aggregate information from the earlier characters.

    In:  q, k, v — (B, H, T, d_head) each: queries, keys, values per head.
         mask    — (T, T) boolean, True = allowed to attend (from causal_mask).
    Out: (B, H, T, d_head) — for each head: the values v weighted by softmax
         over allowed positions of (q·k / sqrt(d_head)).
    """
    d_head = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_head ** 0.5)
    scores = scores.masked_fill(~mask, float('-inf'))
    attn_weights = softmax(scores)
    return torch.matmul(attn_weights, v)


def generate(model, start_ids, max_new_tokens, temperature, top_p):
    """Write one character at a time, using temperature and top-p.

    In:  model          — a GPT from model.py; model(ids) gives logits (1, t, V),
                          and model.block_size is the longest context it takes.
         start_ids      — (1, t0) integer ids to start from (already on the
                          model's device).
         max_new_tokens — how many characters to add.
         temperature    — float > 0, divides the scores before softmax.
         top_p          — float in (0, 1]; keep the most likely characters until
                          their probabilities add up to top_p, drop the rest,
                          then sample from what is left. top_p = 1 keeps all.
    Out: (1, t0 + max_new_tokens) — the start ids with the new ids appended.
    """
    for _ in range(max_new_tokens):
        context_ids = start_ids[:, -model.block_size:]
        logits = model(context_ids)
        next_logits = logits[:, -1, :] / temperature
        probs = softmax(next_logits)
        
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            probs = probs.masked_fill(indices_to_remove, 0.0)
            probs = probs / torch.sum(probs, dim=-1, keepdim=True)
            
        next_id = torch.multinomial(probs, num_samples=1)
        start_ids = torch.cat([start_ids, next_id], dim=1)
        
    return start_ids



