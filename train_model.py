import torch
from model import Transformer
from utils import (
    BATCH_SIZE,
    BLOCK_SIZE,
    DEVICE,
    DROPOUT,
    LEARNING_RATE,
    NUM_EMBED,
    NUM_HEAD,
    NUM_LAYER,
    encode,
    decode,
    build_vocab,
    get_batch,
    save_model_to_chekpoint,
)

# raw data
path_do_data = "data/english.txt"
data_raw = open(path_do_data, encoding="utf-8").read()
vocab, vocab_size = build_vocab(path_to_data="data/english.txt")

# load model from checkpoint
# m = load_model_from_checkpoint(Transformer,vocab_size=vocab_size)

# example to decode sequence
# enc_sec = m.generate(idx=torch.zeros((1,1), dtype=torch.long),
# max_new_tokens=20)[0].tolist()
# print(decode(vocab=vocab, enc_sec=enc_sec))

# train a new model
train_data_raw = data_raw[-10004:]
val_data_raw = data_raw[-20004:-10004]
train_data = torch.tensor(encode(vocab, train_data_raw), dtype=torch.long)
val_data = torch.tensor(encode(vocab, val_data_raw), dtype=torch.long)
model = Transformer(
    vocab_size=vocab_size,
    num_embed=NUM_EMBED,
    block_size=BLOCK_SIZE,
    num_heads=NUM_HEAD,
    num_layers=NUM_LAYER,
    dropout=DROPOUT,
)
# load model to GPU if available
m = model.to(DEVICE)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")
# optimizer takes the model's parameters and the learning rate as input,
# and updates the parameters during the training process in order to
# minimize the loss function.
optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)

for step in range(10):
    # sample a batch of data
    xb, yb = get_batch(data=train_data, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE)
    logits, loss = m.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % 500 == 0:
        print("Step: {:6} | Loss: {:6.2f}".format(step, loss.item()))

save_model_to_chekpoint(model=m, path_to_checkpoint="checkpoints")

context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))
