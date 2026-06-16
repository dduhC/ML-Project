# from sentence_transformers import SentenceTransformer, InputExample, losses
# from torch.utils.data import DataLoader

# model = SentenceTransformer('all-MiniLM-L6-v2')

# train_examples = [
#     InputExample(texts=[anchor, positive, negative]),
#     # ...
# ]

# train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
# train_loss = losses.TripletLoss(model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=0.5)

# model.fit(
#     train_objectives=[(train_dataloader, train_loss)],
#     epochs=3,
#     warmup_steps=100,
#     output_path='./finetuned-arxiv-sbert'
# )

from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
from sentence_transformers.util import batch_to_device
import torch

model = SentenceTransformer("all-MiniLM-L6-v2")
device = model.device

anchors = ["It's nice weather outside today.", "He drove to work."]
positives = ["It's so sunny.", "He took the car to the office."]
negatives = ["It's quite rainy, sadly.", "She walked to the store."]

# Tokenize từng phần
features_anchor = batch_to_device(model.preprocess(anchors), device)
features_positive = batch_to_device(model.preprocess(positives), device)
features_negative = batch_to_device(model.preprocess(negatives), device)

sentence_features = [features_anchor, features_positive, features_negative]

loss_fn = losses.TripletLoss(model=model)

# labels không dùng cho TripletLoss nhưng API yêu cầu vẫn pass vào
loss_value = loss_fn(sentence_features, labels=None)

print(loss_value.item())