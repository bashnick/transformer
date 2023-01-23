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
    MAX_ITER,
    EVAL_INTER,
    encode,
    decode,
    build_vocab,
    get_batch,
    save_model_to_chekpoint,
    estimate_loss,
)

# raw data
path_do_data = "data/english.txt"
data_raw = open(path_do_data, encoding="utf-8").read()
data_raw = data_raw[4000000:]  # short dataset
vocab, vocab_size = build_vocab(path_to_data="data/english.txt")

# load model from checkpoint
# m = load_model_from_checkpoint(Transformer,vocab_size=vocab_size)

# example to decode sequence
# enc_sec = m.generate(idx=torch.zeros((1,1), dtype=torch.long),
# max_new_tokens=20)[0].tolist()
# print(decode(vocab=vocab, enc_sec=enc_sec))

# train/val split
data = torch.tensor(encode(vocab, data_raw), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# train a new model
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
print(
    "Model with {:.2f}M parameters".format(sum(p.numel() for p in m.parameters()) / 1e6)
)
# optimizer takes the model's parameters and the learning rate as input,
# and updates the parameters during the training process in order to
# minimize the loss function.
optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)

for step in range(MAX_ITER):

    # every EVAL_INTER evaluate the loss on train and val sets
    if iter % EVAL_INTER == 0 or iter == MAX_ITER - 1:
        loss_train = estimate_loss(
            data=train_data, model=m, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE
        )
        loss_val = estimate_loss(
            data=val_data, model=m, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE
        )
        print(
            f"step {iter:10} | train loss {loss_train:6.4f} | val loss {loss_val:6.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch(data=train_data, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE)
    logits, loss = m.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    # if step % 500 == 0:
    #     print("Step: {:6} | Loss: {:6.2f}".format(step, loss.item()))

save_model_to_chekpoint(model=m, path_to_checkpoint="checkpoints", epoch=step)

context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(
    decode(
        vocab=vocab,
        enc_sec=m.generate(context, max_new_tokens=100, block_size=BLOCK_SIZE)[
            0
        ].tolist(),
    )
)
