from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd

# Load data
triplets_df = pd.read_csv("finetune_BERT/triplets.csv")
df = pd.read_csv("finetune_BERT/train_content.csv").reset_index(drop=True)

# Map paper_id → text
id_to_text = dict(zip(df["paper_id"], df["content"]))  # hoặc df["title"] + " " + df["abstract"]

# Build InputExample list
train_examples = []
for _, row in triplets_df.iterrows():
    anchor   = id_to_text.get(row["anchor_id"])
    positive = id_to_text.get(row["positive_id"])
    negative = id_to_text.get(row["negative_id"])
    
    if anchor and positive and negative:  # skip nếu có id bị missing
        train_examples.append(InputExample(texts=[anchor, positive, negative]))

print(f"Training examples: {len(train_examples)}")

# Train
model = SentenceTransformer('all-MiniLM-L6-v2')
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.TripletLoss(
    model,
    distance_metric=losses.TripletDistanceMetric.COSINE,
    triplet_margin=0.5
)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path='finetune_BERT/model_checkpoint/finetuned-arxiv-sbert'
)