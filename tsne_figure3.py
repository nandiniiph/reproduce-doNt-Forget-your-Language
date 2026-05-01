import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModel
import random

# =========================
# DEVICE
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# =========================
# LOAD MODELS
# =========================
model_co = AutoModel.from_pretrained(
    "outputs/amazon/nfl_co/checkpoint-7002"
).to(device)

model_cp = AutoModel.from_pretrained(
    "outputs/amazon/nfl_cp/checkpoint-7002"
).to(device)

model_co.eval()
model_cp.eval()

# =========================
# DATA
# =========================
sentences = {
    "irrelevant": [
        "the sky is blue",
        "people are walking",
        "i saw a cat",
        "weather is nice"
    ],
    "positive": [
        "this is good",
        "i love this",
        "amazing product"
    ],
    "negative": [
        "this is bad",
        "i hate this",
        "terrible product"
    ],
    "book": [
        "this book is good",
        "i read a book"
    ],
    "movie": [
        "this movie is good",
        "i watched a movie"
    ]
}

texts = []
labels = []

for label, sents in sentences.items():
    for _ in range(120):
        texts.append(random.choice(sents))
        labels.append(label)

# =========================
# EMBEDDING
# =========================
def get_emb(model, texts):
    embs = []
    for t in texts:
        inputs = tokenizer(t, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            out = model(**inputs)
        cls = out.last_hidden_state[:, 0, :].cpu().numpy()[0]
        embs.append(cls)
    return np.array(embs)

print("Extract embeddings...")

X_co = get_emb(model_co, texts)
X_cp = get_emb(model_cp, texts)

# =========================
# TSNE
# =========================
print("Running TSNE...")

tsne = TSNE(n_components=2, perplexity=40, n_iter=2000, random_state=42)

Z_co = tsne.fit_transform(X_co)
Z_cp = tsne.fit_transform(X_cp)

# =========================
# PLOT
# =========================
def plot(ax, Z, labels, title):
    style = {
        "irrelevant": ("gray", "o"),
        "positive": ("blue", "+"),
        "negative": ("red", "x"),
        "book": ("green", "*"),
        "movie": ("orange", "X")
    }

    for l in set(labels):
        idx = [i for i, x in enumerate(labels) if x == l]
        color, marker = style[l]

        ax.scatter(
            Z[idx, 0],
            Z[idx, 1],
            c=color,
            marker=marker,
            label=l,
            alpha=0.7
        )

    ax.set_title(title)
    ax.legend()

# =========================
# FIGURE
# =========================
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

plot(axs[0], Z_co, labels, "(a) NFL-CO")
plot(axs[1], Z_cp, labels, "(b) NFL-CP")

plt.tight_layout()
plt.savefig("figure3_tsne.png", dpi=300)
plt.show()

print("Saved as figure3_tsne.png")