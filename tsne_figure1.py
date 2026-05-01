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
# LOAD MODELS
# =========================
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# before fine-tuning
model_pre = AutoModel.from_pretrained("roberta-base").to(device)

# after fine-tuning (ganti path sesuai punyamu)
model_ft = AutoModel.from_pretrained(
    "outputs/amazon/none/checkpoint-7002"
).to(device)

model_pre.eval()
model_ft.eval()

# =========================
# DATASET (PENTING)
# =========================
sentences = {
    "irrelevant": [
        "the sky is blue today",
        "people are walking outside",
        "i saw a cat yesterday",
        "weather is nice today",
        "this is something random"
    ],
    "positive": [
        "this product is good",
        "i love this movie",
        "this is amazing",
        "very nice experience"
    ],
    "negative": [
        "this product is bad",
        "i hate this movie",
        "this is terrible",
        "very bad experience"
    ],
    "book": [
        "this book is good",
        "i read a good book"
    ],
    "movie": [
        "this movie is good",
        "i watched a good movie"
    ]
}

texts = []
labels = []

# generate banyak sample biar cluster jelas
for label, sents in sentences.items():
    for _ in range(120):
        texts.append(random.choice(sents))
        labels.append(label)

# =========================
# EMBEDDING FUNCTION
# =========================
def get_embeddings(model, texts):
    embs = []
    for t in texts:
        inputs = tokenizer(t, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            out = model(**inputs)
        cls_vec = out.last_hidden_state[:, 0, :].cpu().numpy()[0]
        embs.append(cls_vec)
    return np.array(embs)

print("Extracting embeddings...")

X_pre = get_embeddings(model_pre, texts)
X_ft = get_embeddings(model_ft, texts)

# =========================
# TSNE
# =========================
print("Running t-SNE...")

tsne = TSNE(
    n_components=2,
    perplexity=40,
    n_iter=2000,
    random_state=42
)

Z_pre = tsne.fit_transform(X_pre)
Z_ft = tsne.fit_transform(X_ft)

# =========================
# PLOT FUNCTION
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
# PLOT FIGURE
# =========================
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

plot(axs[0], Z_pre, labels, "(a) Initial")
plot(axs[1], Z_ft, labels, "(b) Standard fine-tuning")

plt.tight_layout()
plt.savefig("figure1_tsne.png", dpi=300)
plt.show()

print("Saved as figure1_tsne.png")