import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD MODEL
# =========================
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

model_pre = AutoModel.from_pretrained("roberta-base").to(device)
model_ft = AutoModel.from_pretrained(
    "outputs/amazon/none/checkpoint-7002"
).to(device)

model_pre.eval()
model_ft.eval()

# =========================
# VOCAB SAMPLE
# =========================
words = [
    "movie", "book", "film", "music", "story", "library",
    "good", "bad", "amazing", "terrible",
    "happy", "sad", "love", "hate",
    "internet", "magic", "picture"
]

# =========================
# EMBEDDING FUNCTION
# =========================
def get_emb(model, word):
    inputs = tokenizer(word, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state[:,0,:].cpu().numpy()[0]

# =========================
# BUILD EMBEDDINGS
# =========================
emb_pre = {w: get_emb(model_pre, w) for w in words}
emb_ft = {w: get_emb(model_ft, w) for w in words}

# =========================
# FIND NEIGHBORS
# =========================
def get_neighbors(target, emb_dict, topk=5):
    sims = []
    target_vec = emb_dict[target]

    for w, vec in emb_dict.items():
        if w == target:
            continue
        sim = cosine_similarity([target_vec], [vec])[0][0]
        sims.append((w, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    return [w for w,_ in sims[:topk]]

# =========================
# TARGET TOKENS
# =========================
targets = ["movie", "book"]

print("\n===== NEAREST NEIGHBORS =====\n")

for t in targets:
    before = get_neighbors(t, emb_pre)
    after = get_neighbors(t, emb_ft)

    print(f"Target: {t}")
    print(f"Before: {before}")
    print(f"After : {after}")
    print("-"*40)