from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd, numpy as np

pretrained  = SentenceTransformer('all-MiniLM-L6-v2')
finetuned_v1   = SentenceTransformer('finetune_BERT/model_checkpoint/finetuned-arxiv-sbert_v1')
finetuned_v2   = SentenceTransformer('finetune_BERT/model_checkpoint/finetuned-arxiv-sbert_v2')

# Lấy vài triplet từ triplets.csv để test
triplets_df = pd.read_csv("finetune_BERT/triplets_18k.csv").sample(200, random_state=42)
df          = pd.read_csv("data/train.csv").set_index("paper_id")
id_to_text  = df["abstract"].to_dict()

results = []
for _, row in triplets_df.iterrows():
    a = id_to_text.get(row["anchor_id"])
    p = id_to_text.get(row["positive_id"])
    n = id_to_text.get(row["negative_id"])
    if not (a and p and n): continue

    for name, m in [("pretrained", pretrained), ("finetuned_v1", finetuned_v1), ("finetuned_v2", finetuned_v2)]:
        ea, ep, en = m.encode([a, p, n])
        sim_pos = cosine_similarity([ea], [ep])[0][0]
        sim_neg = cosine_similarity([ea], [en])[0][0]
        results.append({"model": name, "sim_pos": sim_pos, "sim_neg": sim_neg, "margin": sim_pos - sim_neg})

res = pd.DataFrame(results)
print(res.groupby("model")[["sim_pos", "sim_neg", "margin"]].mean())