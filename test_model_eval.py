#!/usr/bin/env python3
"""Test if model eval forward pass works."""
import torch
from train import GPTConfig, GPT
from prepare import Tokenizer, make_dataloader, MAX_SEQ_LEN

print("Setting up device...", flush=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

print("Loading tokenizer...", flush=True)
tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()

print(f"Creating model on {device}...", flush=True)
config = GPTConfig(
    sequence_len=MAX_SEQ_LEN,
    vocab_size=vocab_size,
    n_layer=10,
    n_head=10,
    n_kv_head=1,
    n_embd=640,
    window_pattern="L"
)
model = GPT(config).to(device).to(dtype)
model = torch.compile(model, dynamic=False)

print("Creating val dataloader...", flush=True)
val_loader = make_dataloader(tokenizer, 4, MAX_SEQ_LEN, "val")

print("Switching to eval mode...", flush=True)
model.eval()

print("Fetching val batch...", flush=True)
x, y, _ = next(val_loader)
x = x.to(device)
y = y.to(device)

print(f"Input shapes: x={x.shape}, y={y.shape}", flush=True)

print("Running model forward with reduction='none'...", flush=True)
with torch.no_grad():
    loss = model(x, y, reduction='none')
    print(f"Loss shape: {loss.shape}, dtype: {loss.dtype}", flush=True)
    loss_flat = loss.view(-1)
    print(f"Loss flat shape: {loss_flat.shape}", flush=True)

print("Test complete - model eval works!", flush=True)
