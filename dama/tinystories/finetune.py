import torch
from torch.distributions import Categorical
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from datasets import load_dataset
import genki

NUM_ITERS = 1000
BATCH_SIZE = 1024
BLOCK_SIZE = 16
LR = 1e-5
DEVICE = "mps"  # "cpu", "mps", "cuda" or "crynux"

model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
model = genki.to(model, DEVICE)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_tokens = tokenizer.eos_token

# Data process
with open('tiny_shakespeare.txt', 'r', encoding='utf-8') as f:
    corpus = f.read()
data = tokenizer.encode(corpus, return_tensors="pt")[0]
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]
print(len(train_data), len(val_data))


def get_data(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    x, y = genki.to(x, DEVICE), genki.to(y, DEVICE)
    return x, y


optimizer = AdamW(model.parameters(), lr=LR)
model.train()

for iters in range(NUM_ITERS):
    x, y = get_data("train")
    optimizer.zero_grad()
    outputs = genki.call(model, input_ids=x, labels=y)
    loss = outputs[0]
    loss.backward()
    optimizer.step()
    print(f"Iteration: {iters}, Loss: {loss}")

save_directory = "./checkpoints"
model.save_pretrained(save_directory)
