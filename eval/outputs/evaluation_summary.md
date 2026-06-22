# Representation Evaluation Summary

Protocol: 500 synthetic test users, hybrid alpha=0.7, profile half-life=90 days, seed=42.

| representation | k | category_precision_mean | ndcg_mean | hit_rate_mean | semantic_coherence_mean | ild_mean |
|---|---|---|---|---|---|---|
| TF-IDF | 5 | 0.8276 | 0.0 | 0.0 | 0.2751 | 0.7934 |
| TF-IDF | 10 | 0.8164 | 0.0006 | 0.002 | 0.2662 | 0.8106 |
| TF-IDF | 20 | 0.8068 | 0.0011 | 0.004 | 0.2563 | 0.8299 |
| SBERT pretrained | 5 | 0.8848 | 0.0 | 0.0 | 0.7305 | 0.383 |
| SBERT pretrained | 10 | 0.8796 | 0.0 | 0.0 | 0.7229 | 0.396 |
| SBERT pretrained | 20 | 0.872 | 0.0 | 0.0 | 0.7138 | 0.4095 |
| SBERT fine-tuned | 5 | 1.0 | 0.0 | 0.0 | 0.7997 | 0.1696 |
| SBERT fine-tuned | 10 | 1.0 | 0.0 | 0.0 | 0.791 | 0.1826 |
| SBERT fine-tuned | 20 | 1.0 | 0.0 | 0.0 | 0.7803 | 0.2008 |

## Silhouette scores

| Representation | Silhouette (cosine) |
|---|---:|
| TF-IDF | 0.0153 |
| SBERT pretrained | 0.0467 |
| SBERT fine-tuned | 0.3749 |

t-SNE is a qualitative visualization. Ranking metrics are the primary evidence for recommendation quality.