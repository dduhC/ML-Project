---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:89059
- loss:TripletLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: 'Pooled Motion Features for First-Person Videos cs.CV In this paper,
    we present a new feature representation for first-person

    videos. In first-person video understanding (e.g., activity recognition), it is

    very important to capture both entire scene dynamics (i.e., egomotion) and

    salient local motion observed in videos. We describe a representation framework

    based on time series pooling, which is designed to abstract

    short-term/long-term changes in feature descriptor elements. The idea is to

    keep track of how descriptor values are changing over time and summarize them

    to represent motion in the activity video. The framework is general, handling

    any types of per-frame feature descriptors including conventional motion

    descriptors like histogram of optical flows (HOF) as well as appearance

    descriptors from more recent convolutional neural networks (CNN). We

    experimentally confirm that our approach clearly outperforms previous feature

    representations including bag-of-visual-words and improved Fisher vector (IFV)

    when using identical underlying feature descriptors. We also confirm that our

    feature representation has superior performance to existing state-of-the-art

    features like local spatio-temporal features and Improved Trajectory Features

    (originally developed for 3rd-person videos) when handling first-person videos.

    Multiple first-person activity datasets were tested under various settings to

    confirm these findings.'
  sentences:
  - 'Face Transformer: Towards High Fidelity and Accurate Face Swapping cs.CV Face
    swapping aims to generate swapped images that fuse the identity of

    source faces and the attributes of target faces. Most existing works address

    this challenging task through 3D modelling or generation using generative

    adversarial networks (GANs), but 3D modelling suffers from limited

    reconstruction accuracy and GANs often struggle in preserving subtle yet

    important identity details of source faces (e.g., skin colors, face features)

    and structural attributes of target faces (e.g., face shapes, facial

    expressions). This paper presents Face Transformer, a novel face swapping

    network that can accurately preserve source identities and target attributes

    simultaneously in the swapped face images. We introduce a transformer network

    for the face swapping task, which learns high-quality semantic-aware

    correspondence between source and target faces and maps identity features of

    source faces to the corresponding region in target faces. The high-quality

    semantic-aware correspondence enables smooth and accurate transfer of source

    identity information with minimal modification of target shapes and

    expressions. In addition, our Face Transformer incorporates a multi-scale

    transformation mechanism for preserving the rich fine facial details. Extensive

    experiments show that our Face Transformer achieves superior face swapping

    performance qualitatively and quantitatively.'
  - 'Data Movement Is All You Need: A Case Study on Optimizing Transformers cs.LG
    Transformers are one of the most important machine learning workloads today.

    Training one is a very compute-intensive task, often taking days or weeks, and

    significant attention has been given to optimizing transformers. Despite this,

    existing implementations do not efficiently utilize GPUs. We find that data

    movement is the key bottleneck when training. Due to Amdahl''s Law and massive

    improvements in compute performance, training has now become memory-bound.

    Further, existing frameworks use suboptimal data layouts. Using these insights,

    we present a recipe for globally optimizing data movement in transformers. We

    reduce data movement by up to 22.91% and overall achieve a 1.30x performance

    improvement over state-of-the-art frameworks when training a BERT encoder layer

    and 1.19x for the entire BERT. Our approach is applicable more broadly to

    optimizing deep neural networks, and offers insight into how to tackle emerging

    performance bottlenecks.'
  - 'Three-Stream Fusion Network for First-Person Interaction Recognition cs.CV First-person
    interaction recognition is a challenging task because of

    unstable video conditions resulting from the camera wearer''s movement. For

    human interaction recognition from a first-person viewpoint, this paper

    proposes a three-stream fusion network with two main parts: three-stream

    architecture and three-stream correlation fusion. Thre three-stream

    architecture captures the characteristics of the target appearance, target

    motion, and camera ego-motion. Meanwhile the three-stream correlation fusion

    combines the feature map of each of the three streams to consider the

    correlations among the target appearance, target motion and camera ego-motion.

    The fused feature vector is robust to the camera movement and compensates for

    the noise of the camera ego-motion. Short-term intervals are modeled using the

    fused feature vector, and a long short-term memory(LSTM) model considers the

    temporal dynamics of the video. We evaluated the proposed method on two-public

    benchmark datasets to validate the effectiveness of our approach. The

    experimental results show that the proposed fusion method successfully

    generated a discriminative feature vector, and our network outperformed all

    competing activity recognition methods in first-person videos where

    considerable camera ego-motion occurs.'
- source_sentence: "Resonate-and-Fire Spiking Neurons for Target Detection and Hand\
    \ Gesture\n  Recognition: A Hybrid Approach eess.SP Hand gesture recognition using\
    \ radar often relies on computationally\nexpensive fast Fourier transforms. This\
    \ paper proposes an alternative approach\nthat bypasses fast Fourier transforms\
    \ using resonate-and-fire neurons. These\nneurons directly detect the hand in\
    \ the time-domain signal, eliminating the\nneed for fast Fourier transforms to\
    \ retrieve range information. Following\ndetection, a simple Goertzel algorithm\
    \ is employed to extract five key\nfeatures, eliminating the need for a second\
    \ fast Fourier transform. These\nfeatures are then fed into a recurrent neural\
    \ network, achieving an accuracy of\n98.21% for classifying five gestures. The\
    \ proposed approach demonstrates\ncompetitive performance with reduced complexity\
    \ compared to traditional methods"
  sentences:
  - "Analysis of False Data Injection Impact on AI based Solar Photovoltaic\n  Power\
    \ Generation Forecasting eess.SP The use of solar photovoltaics (PV) energy provides\
    \ additional resources to\nthe electric power grid. The downside of this integration\
    \ is that the solar\npower supply is unreliable and highly dependent on the weather\
    \ condition. The\npredictability and stability of forecasting are critical for\
    \ the full\nutilization of solar power. This study reviews and evaluates various\
    \ machine\nlearning-based models for solar PV power generation forecasting using\
    \ a public\ndataset. Furthermore, The root mean squared error (RMSE), mean squared\
    \ error\n(MSE), and mean average error (MAE) metrics are used to evaluate the\
    \ results.\nLinear Regression, Gaussian Process Regression, K-Nearest Neighbor,\
    \ Decision\nTrees, Gradient Boosting Regression Trees, Multi-layer Perceptron,\
    \ and Support\nVector Regression algorithms are assessed. Their responses against\
    \ false data\ninjection attacks are also investigated. The Multi-layer Perceptron\
    \ Regression\nmethod shows robust prediction on both regular and noise injected\
    \ datasets over\nother methods."
  - "MEVID: Multi-view Extended Videos with Identities for Video Person\n  Re-Identification\
    \ cs.CV In this paper, we present the Multi-view Extended Videos with Identities\n\
    (MEVID) dataset for large-scale, video person re-identification (ReID) in the\n\
    wild. To our knowledge, MEVID represents the most-varied video person ReID\ndataset,\
    \ spanning an extensive indoor and outdoor environment across nine\nunique dates\
    \ in a 73-day window, various camera viewpoints, and entity clothing\nchanges.\
    \ Specifically, we label the identities of 158 unique people wearing 598\noutfits\
    \ taken from 8, 092 tracklets, average length of about 590 frames, seen\nin 33\
    \ camera views from the very large-scale MEVA person activities dataset.\nWhile\
    \ other datasets have more unique identities, MEVID emphasizes a richer set\n\
    of information about each individual, such as: 4 outfits/identity vs. 2\noutfits/identity\
    \ in CCVID, 33 viewpoints across 17 locations vs. 6 in 5\nsimulated locations\
    \ for MTA, and 10 million frames vs. 3 million for LS-VID.\nBeing based on the\
    \ MEVA video dataset, we also inherit data that is\nintentionally demographically\
    \ balanced to the continental United States. To\naccelerate the annotation process,\
    \ we developed a semi-automatic annotation\nframework and GUI that combines state-of-the-art\
    \ real-time models for object\ndetection, pose estimation, person ReID, and multi-object\
    \ tracking. We evaluate\nseveral state-of-the-art methods on MEVID challenge problems\
    \ and\ncomprehensively quantify their robustness in terms of changes of outfit,\
    \ scale,\nand background location. Our quantitative analysis on the realistic,\
    \ unique\naspects of MEVID shows that there are significant remaining challenges\
    \ in video\nperson ReID and indicates important directions for future research."
  - "Transformer-based Hand Gesture Recognition via High-Density EMG Signals:\n  From\
    \ Instantaneous Recognition to Fusion of Motor Unit Spike Trains eess.SP Designing\
    \ efficient and labor-saving prosthetic hands requires powerful hand\ngesture\
    \ recognition algorithms that can achieve high accuracy with limited\ncomplexity\
    \ and latency. In this context, the paper proposes a compact deep\nlearning framework\
    \ referred to as the CT-HGR, which employs a vision\ntransformer network to conduct\
    \ hand gesture recognition using highdensity sEMG\n(HD-sEMG) signals. The attention\
    \ mechanism in the proposed model identifies\nsimilarities among different data\
    \ segments with a greater capacity for parallel\ncomputations and addresses the\
    \ memory limitation problems while dealing with\ninputs of large sequence lengths.\
    \ CT-HGR can be trained from scratch without\nany need for transfer learning and\
    \ can simultaneously extract both temporal and\nspatial features of HD-sEMG data.\
    \ Additionally, the CT-HGR framework can\nperform instantaneous recognition using\
    \ sEMG image spatially composed from\nHD-sEMG signals. A variant of the CT-HGR\
    \ is also designed to incorporate\nmicroscopic neural drive information in the\
    \ form of Motor Unit Spike Trains\n(MUSTs) extracted from HD-sEMG signals using\
    \ Blind Source Separation (BSS).\nThis variant is combined with its baseline version\
    \ via a hybrid architecture to\nevaluate potentials of fusing macroscopic and\
    \ microscopic neural drive\ninformation. The utilized HD-sEMG dataset involves\
    \ 128 electrodes that collect\nthe signals related to 65 isometric hand gestures\
    \ of 20 subjects. The proposed\nCT-HGR framework is applied to 31.25, 62.5, 125,\
    \ 250 ms window sizes of the\nabove-mentioned dataset utilizing 32, 64, 128 electrode\
    \ channels. The average\naccuracy over all the participants using 32 electrodes\
    \ and a window size of\n31.25 ms is 86.23%, which gradually increases till reaching\
    \ 91.98% for 128\nelectrodes and a window size of 250 ms. The CT-HGR achieves\
    \ accuracy of 89.13%\nfor instantaneous recognition based on a single frame of\
    \ HD-sEMG image."
- source_sentence: 'Prompts have evil twins cs.CL We discover that many natural-language
    prompts can be replaced by

    corresponding prompts that are unintelligible to humans but that provably

    elicit similar behavior in language models. We call these prompts "evil twins"

    because they are obfuscated and uninterpretable (evil), but at the same time

    mimic the functionality of the original natural-language prompts (twins).

    Remarkably, evil twins transfer between models. We find these prompts by

    solving a maximum-likelihood problem which has applications of independent

    interest.'
  sentences:
  - 'Semantic Tagging with Deep Residual Networks cs.CL We propose a novel semantic
    tagging task, sem-tagging, tailored for the

    purpose of multilingual semantic parsing, and present the first tagger using

    deep residual networks (ResNets). Our tagger uses both word and character

    representations and includes a novel residual bypass architecture. We evaluate

    the tagset both intrinsically on the new task of semantic tagging, as well as

    on Part-of-Speech (POS) tagging. Our system, consisting of a ResNet and an

    auxiliary loss function predicting our semantic tags, significantly outperforms

    prior results on English Universal Dependencies POS tagging (95.71% accuracy on

    UD v1.2 and 95.67% accuracy on UD v1.3).'
  - "Zero-Shot Continuous Prompt Transfer: Generalizing Task Semantics Across\n  Language\
    \ Models cs.CL Prompt tuning in natural language processing (NLP) has become an\
    \ increasingly\npopular method for adapting large language models to specific\
    \ tasks. However,\nthe transferability of these prompts, especially continuous\
    \ prompts, between\ndifferent models remains a challenge. In this work, we propose\
    \ a zero-shot\ncontinuous prompt transfer method, where source prompts are encoded\
    \ into\nrelative space and the corresponding target prompts are searched for\n\
    transferring to target models. Experimental results confirm the effectiveness\n\
    of our method, showing that 'task semantics' in continuous prompts can be\ngeneralized\
    \ across various language models. Moreover, we find that combining\n'task semantics'\
    \ from multiple source models can further enhance the\ngeneralizability of transfer."
  - 'Training Discriminative Models to Evaluate Generative Ones cs.LG Generative models
    are known to be difficult to assess. Recent works,

    especially on generative adversarial networks (GANs), produce good visual

    samples of varied categories of images. However, the validation of their

    quality is still difficult to define and there is no existing agreement on the

    best evaluation process. This paper aims at making a step toward an objective

    evaluation process for generative models. It presents a new method to assess a

    trained generative model by evaluating the test accuracy of a classifier

    trained with generated data. The test set is composed of real images.

    Therefore, The classifier accuracy is used as a proxy to evaluate if the

    generative model fit the true data distribution. By comparing results with

    different generated datasets we are able to classify and compare generative

    models. The motivation of this approach is also to evaluate if generative

    models can help discriminative neural networks to learn, i.e., measure if

    training on generated data is able to make a model successful at testing on

    real settings. Our experiments compare different generators from the

    Variational Auto-Encoders (VAE) and Generative Adversarial Network (GAN)

    frameworks on MNIST and fashion MNIST datasets. Our results show that none of

    the generative models is able to replace completely true data to train a

    discriminative model. But they also show that the initial GAN and WGAN are the

    best choices to generate on MNIST database (Modified National Institute of

    Standards and Technology database) and fashion MNIST database.'
- source_sentence: "Programming with Personalized PageRank: A Locally Groundable First-Order\n\
    \  Probabilistic Logic cs.AI In many probabilistic first-order representation\
    \ systems, inference is\nperformed by \"grounding\"---i.e., mapping it to a propositional\
    \ representation,\nand then performing propositional inference. With a large database\
    \ of facts,\ngroundings can be very large, making inference and learning computationally\n\
    expensive. Here we present a first-order probabilistic language which is\nwell-suited\
    \ to approximate \"local\" grounding: every query $Q$ can be\napproximately grounded\
    \ with a small graph. The language is an extension of\nstochastic logic programs\
    \ where inference is performed by a variant of\npersonalized PageRank. Experimentally,\
    \ we show that the approach performs well\nwithout weight learning on an entity\
    \ resolution task; that supervised\nweight-learning improves accuracy; and that\
    \ grounding time is independent of DB\nsize. We also show that order-of-magnitude\
    \ speedups are possible by\nparallelizing learning."
  sentences:
  - "Constructing Confidence Intervals for 'the' Generalization Error -- a\n  Comprehensive\
    \ Benchmark Study stat.ML When assessing the quality of prediction models in machine\
    \ learning,\nconfidence intervals (CIs) for the generalization error, which measures\n\
    predictive performance, are a crucial tool. Luckily, there exist many methods\n\
    for computing such CIs and new promising approaches are continuously being\nproposed.\
    \ Typically, these methods combine various resampling procedures, most\npopular\
    \ among them cross-validation and bootstrapping, with different variance\nestimation\
    \ techniques. Unfortunately, however, there is currently no consensus\non when\
    \ any of these combinations may be most reliably employed and how they\ngenerally\
    \ compare. In this work, we conduct a large-scale study comparing CIs\nfor the\
    \ generalization error, the first one of such size, where we empirically\nevaluate\
    \ 13 different CI methods on a total of 19 tabular regression and\nclassification\
    \ problems, using seven different inducers and a total of eight\nloss functions.\
    \ We give an overview of the methodological foundations and\ninherent challenges\
    \ of constructing CIs for the generalization error and\nprovide a concise review\
    \ of all 13 methods in a unified framework. Finally, the\nCI methods are evaluated\
    \ in terms of their relative coverage frequency, width,\nand runtime. Based on\
    \ these findings, we can identify a subset of methods that\nwe would recommend.\
    \ We also publish the datasets as a benchmarking suite on\nOpenML and our code\
    \ on GitHub to serve as a basis for further studies."
  - "Hybridization of Expectation-Maximization and K-Means Algorithms for\n  Better\
    \ Clustering Performance cs.LG The present work proposes hybridization of Expectation-Maximization\
    \ (EM) and\nK-Means techniques as an attempt to speed-up the clustering process.\
    \ Though\nboth K-Means and EM techniques look into different areas, K-means can\
    \ be viewed\nas an approximate way to obtain maximum likelihood estimates for\
    \ the means.\nAlong with the proposed algorithm for hybridization, the present\
    \ work also\nexperiments with the Standard EM algorithm. Six different datasets\
    \ are used for\nthe experiments of which three are synthetic datasets. Clustering\
    \ fitness and\nSum of Squared Errors (SSE) are computed for measuring the clustering\n\
    performance. In all the experiments it is observed that the proposed algorithm\n\
    for hybridization of EM and K-Means techniques is consistently taking less\nexecution\
    \ time with acceptable Clustering Fitness value and less SSE than the\nstandard\
    \ EM algorithm. It is also observed that the proposed algorithm is\nproducing\
    \ better clustering results than the Cluster package of Purdue\nUniversity."
  - "Efficient Inference and Learning in a Large Knowledge Base: Reasoning\n  with\
    \ Extracted Information using a Locally Groundable First-Order\n  Probabilistic\
    \ Logic cs.AI One important challenge for probabilistic logics is reasoning with\
    \ very large\nknowledge bases (KBs) of imperfect information, such as those produced\
    \ by\nmodern web-scale information extraction systems. One scalability problem\
    \ shared\nby many probabilistic logics is that answering queries involves \"grounding\"\
    \ the\nquery---i.e., mapping it to a propositional representation---and the size\
    \ of a\n\"grounding\" grows with database size. To address this bottleneck, we\
    \ present a\nfirst-order probabilistic language called ProPPR in which that approximate\n\
    \"local groundings\" can be constructed in time independent of database size.\n\
    Technically, ProPPR is an extension to stochastic logic programs (SLPs) that is\n\
    biased towards short derivations; it is also closely related to an earlier\nrelational\
    \ learning algorithm called the path ranking algorithm (PRA). We show\nthat the\
    \ problem of constructing proofs for this logic is related to\ncomputation of\
    \ personalized PageRank (PPR) on a linearized version of the proof\nspace, and\
    \ using on this connection, we develop a proveably-correct approximate\ngrounding\
    \ scheme, based on the PageRank-Nibble algorithm. Building on this, we\ndevelop\
    \ a fast and easily-parallelized weight-learning algorithm for ProPPR. In\nexperiments,\
    \ we show that learning for ProPPR is orders magnitude faster than\nlearning for\
    \ Markov logic networks; that allowing mutual recursion (joint\nlearning) in KB\
    \ inference leads to improvements in performance; and that ProPPR\ncan learn weights\
    \ for a mutually recursive program with hundreds of clauses,\nwhich define scores\
    \ of interrelated predicates, over a KB containing one\nmillion entities."
- source_sentence: 'Towards Zero-resource Cross-lingual Entity Linking cs.CL Cross-lingual
    entity linking (XEL) grounds named entities in a source

    language to an English Knowledge Base (KB), such as Wikipedia. XEL is

    challenging for most languages because of limited availability of requisite

    resources. However, much previous work on XEL has been on simulated settings

    that actually use significant resources (e.g. source language Wikipedia,

    bilingual entity maps, multilingual embeddings) that are unavailable in truly

    low-resource languages. In this work, we first examine the effect of these

    resource assumptions and quantify how much the availability of these resource

    affects overall quality of existing XEL systems. Next, we propose three

    improvements to both entity candidate generation and disambiguation that make

    better use of the limited data we do have in resource-scarce scenarios. With

    experiments on four extremely low-resource languages, we show that our model

    results in gains of 6-23% in end-to-end linking accuracy.'
  sentences:
  - "Co-training Embeddings of Knowledge Graphs and Entity Descriptions for\n  Cross-lingual\
    \ Entity Alignment cs.AI Multilingual knowledge graph (KG) embeddings provide\
    \ latent semantic\nrepresentations of entities and structured knowledge with cross-lingual\n\
    inferences, which benefit various knowledge-driven cross-lingual NLP tasks.\n\
    However, precisely learning such cross-lingual inferences is usually hindered\n\
    by the low coverage of entity alignment in many KGs. Since many multilingual\n\
    KGs also provide literal descriptions of entities, in this paper, we introduce\n\
    an embedding-based approach which leverages a weakly aligned multilingual KG\n\
    for semi-supervised cross-lingual learning using entity descriptions. Our\napproach\
    \ performs co-training of two embedding models, i.e. a multilingual KG\nembedding\
    \ model and a multilingual literal description embedding model. The\nmodels are\
    \ trained on a large Wikipedia-based trilingual dataset where most\nentity alignment\
    \ is unknown to training. Experimental results show that the\nperformance of the\
    \ proposed approach on the entity alignment task improves at\neach iteration of\
    \ co-training, and eventually reaches a stage at which it\nsignificantly surpasses\
    \ previous approaches. We also show that our approach has\npromising abilities\
    \ for zero-shot entity alignment, and cross-lingual KG\ncompletion."
  - 'Zero-shot Neural Transfer for Cross-lingual Entity Linking cs.CL Cross-lingual
    entity linking maps an entity mention in a source language to

    its corresponding entry in a structured knowledge base that is in a different

    (target) language. While previous work relies heavily on bilingual lexical

    resources to bridge the gap between the source and the target languages, these

    resources are scarce or unavailable for many low-resource languages. To address

    this problem, we investigate zero-shot cross-lingual entity linking, in which

    we assume no bilingual lexical resources are available in the source

    low-resource language. Specifically, we propose pivot-based entity linking,

    which leverages information from a high-resource "pivot" language to train

    character-level neural entity linking models that are transferred to the source

    low-resource language in a zero-shot manner. With experiments on 9 low-resource

    languages and transfer through a total of 54 languages, we show that our

    proposed pivot-based framework improves entity linking accuracy 17% (absolute)

    on average over the baseline systems, for the zero-shot scenario. Further, we

    also investigate the use of language-universal phonological representations

    which improves average accuracy (absolute) by 36% when transferring between

    languages that use different scripts.'
  - "SU-RUG at the CoNLL-SIGMORPHON 2017 shared task: Morphological\n  Inflection\
    \ with Attentional Sequence-to-Sequence Models cs.CL This paper describes the\
    \ Stockholm University/University of Groningen\n(SU-RUG) system for the SIGMORPHON\
    \ 2017 shared task on morphological\ninflection. Our system is based on an attentional\
    \ sequence-to-sequence neural\nnetwork model using Long Short-Term Memory (LSTM)\
    \ cells, with joint training of\nmorphological inflection and the inverse transformation,\
    \ i.e. lemmatization and\nmorphological analysis. Our system outperforms the baseline\
    \ with a large\nmargin, and our submission ranks as the 4th best team for the\
    \ track we\nparticipate in (task 1, high-resource)."
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for retrieval.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean', 'include_prompt': True})
  (2): Normalize({})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Towards Zero-resource Cross-lingual Entity Linking cs.CL Cross-lingual entity linking (XEL) grounds named entities in a source\nlanguage to an English Knowledge Base (KB), such as Wikipedia. XEL is\nchallenging for most languages because of limited availability of requisite\nresources. However, much previous work on XEL has been on simulated settings\nthat actually use significant resources (e.g. source language Wikipedia,\nbilingual entity maps, multilingual embeddings) that are unavailable in truly\nlow-resource languages. In this work, we first examine the effect of these\nresource assumptions and quantify how much the availability of these resource\naffects overall quality of existing XEL systems. Next, we propose three\nimprovements to both entity candidate generation and disambiguation that make\nbetter use of the limited data we do have in resource-scarce scenarios. With\nexperiments on four extremely low-resource languages, we show that our model\nresults in gains of 6-23% in end-to-end linking accuracy.',
    'Zero-shot Neural Transfer for Cross-lingual Entity Linking cs.CL Cross-lingual entity linking maps an entity mention in a source language to\nits corresponding entry in a structured knowledge base that is in a different\n(target) language. While previous work relies heavily on bilingual lexical\nresources to bridge the gap between the source and the target languages, these\nresources are scarce or unavailable for many low-resource languages. To address\nthis problem, we investigate zero-shot cross-lingual entity linking, in which\nwe assume no bilingual lexical resources are available in the source\nlow-resource language. Specifically, we propose pivot-based entity linking,\nwhich leverages information from a high-resource "pivot" language to train\ncharacter-level neural entity linking models that are transferred to the source\nlow-resource language in a zero-shot manner. With experiments on 9 low-resource\nlanguages and transfer through a total of 54 languages, we show that our\nproposed pivot-based framework improves entity linking accuracy 17% (absolute)\non average over the baseline systems, for the zero-shot scenario. Further, we\nalso investigate the use of language-universal phonological representations\nwhich improves average accuracy (absolute) by 36% when transferring between\nlanguages that use different scripts.',
    'Co-training Embeddings of Knowledge Graphs and Entity Descriptions for\n  Cross-lingual Entity Alignment cs.AI Multilingual knowledge graph (KG) embeddings provide latent semantic\nrepresentations of entities and structured knowledge with cross-lingual\ninferences, which benefit various knowledge-driven cross-lingual NLP tasks.\nHowever, precisely learning such cross-lingual inferences is usually hindered\nby the low coverage of entity alignment in many KGs. Since many multilingual\nKGs also provide literal descriptions of entities, in this paper, we introduce\nan embedding-based approach which leverages a weakly aligned multilingual KG\nfor semi-supervised cross-lingual learning using entity descriptions. Our\napproach performs co-training of two embedding models, i.e. a multilingual KG\nembedding model and a multilingual literal description embedding model. The\nmodels are trained on a large Wikipedia-based trilingual dataset where most\nentity alignment is unknown to training. Experimental results show that the\nperformance of the proposed approach on the entity alignment task improves at\neach iteration of co-training, and eventually reaches a stage at which it\nsignificantly surpasses previous approaches. We also show that our approach has\npromising abilities for zero-shot entity alignment, and cross-lingual KG\ncompletion.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[ 1.0000,  0.9633, -0.0514],
#         [ 0.9633,  1.0000, -0.0803],
#         [-0.0514, -0.0803,  1.0000]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 89,059 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>sentence_2</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence_0                                                                           | sentence_1                                                                          | sentence_2                                                                           |
  |:---------|:-------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|
  | type     | string                                                                               | string                                                                              | string                                                                               |
  | modality | text                                                                                 | text                                                                                | text                                                                                 |
  | details  | <ul><li>min: 97 tokens</li><li>mean: 219.07 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 25 tokens</li><li>mean: 219.8 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 98 tokens</li><li>mean: 215.69 tokens</li><li>max: 256 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | sentence_2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>On the Complexity of Constrained Determinantal Point Processes cs.DS Determinantal Point Processes (DPPs) are probabilistic models that arise in<br>quantum physics and random matrix theory and have recently found numerous<br>applications in computer science. DPPs define distributions over subsets of a<br>given ground set, they exhibit interesting properties such as negative<br>correlation, and, unlike other models, have efficient algorithms for sampling.<br>When applied to kernel methods in machine learning, DPPs favor subsets of the<br>given data with more diverse features. However, many real-world applications<br>require efficient algorithms to sample from DPPs with additional constraints on<br>the subset, e.g., partition or matroid constraints that are important to ensure<br>priors, resource or fairness constraints on the sampled subset. Whether one can<br>efficiently sample from DPPs in such constrained settings is an important<br>problem that was first raised in a survey of DPPs by \cite{KuleszaTaskar12} and<br>stud...</code>    | <code>An n-ary Constraint for the Stable Marriage Problem cs.DS We present an n-ary constraint for the stable marriage problem. This<br>constraint acts between two sets of integer variables where the domains of<br>those variables represent preferences. Our constraint enforces stability and<br>disallows bigamy. For a stable marriage instance with $n$ men and $n$ women we<br>require only one of these constraints, and the complexity of enforcing<br>arc-consistency is $O(n^2)$ which is optimal in the size of input. Our<br>computational studies show that our n-ary constraint is significantly faster<br>and more space efficient than the encodings presented in \cite{cp01}. We also<br>introduce a new problem to the constraint community, the sex-equal stable<br>marriage problem.</code>                                                                                                                                                                                                                                                                           | <code>Dual-tree $k$-means with bounded iteration runtime cs.DS k-means is a widely used clustering algorithm, but for $k$ clusters and a<br>dataset size of $N$, each iteration of Lloyd's algorithm costs $O(kN)$ time.<br>Although there are existing techniques to accelerate single Lloyd iterations,<br>none of these are tailored to the case of large $k$, which is increasingly<br>common as dataset sizes grow. We propose a dual-tree algorithm that gives the<br>exact same results as standard $k$-means; when using cover trees, we use<br>adaptive analysis techniques to, under some assumptions, bound the<br>single-iteration runtime of the algorithm as $O(N + k log k)$. To our knowledge<br>these are the first sub-$O(kN)$ bounds for exact Lloyd iterations. We then show<br>that this theoretically favorable algorithm performs competitively in practice,<br>especially for large $N$ and $k$ in low dimensions. Further, the algorithm is<br>tree-independent, so any type of tree may be used.</code>                                                            |
  | <code>Argument Invention from First Principles cs.CL Competitive debaters often find themselves facing a challenging task -- how<br>to debate a topic they know very little about, with only minutes to prepare,<br>and without access to books or the Internet? What they often do is rely on<br>"first principles", commonplace arguments which are relevant to many topics,<br>and which they have refined in past debates.<br>  In this work we aim to explicitly define a taxonomy of such principled<br>recurring arguments, and, given a controversial topic, to automatically<br>identify which of these arguments are relevant to the topic.<br>  As far as we know, this is the first time that this approach to argument<br>invention is formalized and made explicit in the context of NLP.<br>  The main goal of this work is to show that it is possible to define such a<br>taxonomy. While the taxonomy suggested here should be thought of as a "first<br>attempt" it is nonetheless coherent, covers well the relevant topics and<br>coincides with what profession...</code> | <code>What if we had no Wikipedia? Domain-independent Term Extraction from a<br>  Large News Corpus cs.CL One of the most impressive human endeavors of the past two decades is the<br>collection and categorization of human knowledge in the free and accessible<br>format that is Wikipedia. In this work we ask what makes a term worthy of<br>entering this edifice of knowledge, and having a page of its own in Wikipedia?<br>To what extent is this a natural product of on-going human discourse and<br>discussion rather than an idiosyncratic choice of Wikipedia editors?<br>Specifically, we aim to identify such "wiki-worthy" terms in a massive news<br>corpus, and see if this can be done with no, or minimal, dependency on actual<br>Wikipedia entries. We suggest a five-step pipeline for doing so, providing<br>baseline results for all five, and the relevant datasets for benchmarking them.<br>Our work sheds new light on the domain-specific Automatic Term Extraction<br>problem, with the problem at hand being a domain-independent variant of it.</code>    | <code>Efficient Inference in Multi-task Cox Process Models stat.ML We generalize the log Gaussian Cox process (LGCP) framework to model multiple<br>correlated point data jointly. The observations are treated as realizations of<br>multiple LGCPs, whose log intensities are given by linear combinations of<br>latent functions drawn from Gaussian process priors. The combination<br>coefficients are also drawn from Gaussian processes and can incorporate<br>additional dependencies. We derive closed-form expressions for the moments of<br>the intensity functions and develop an efficient variational inference<br>algorithm that is orders of magnitude faster than competing deterministic and<br>stochastic approximations of multivariate LGCP, coregionalization models, and<br>multi-task permanental processes. Our approach outperforms these benchmarks in<br>multiple problems, offering the current state of the art in modeling<br>multivariate point processes.</code>                                                                                            |
  | <code>Statistical Depth Functions for Ranking Distributions: Definitions,<br>  Statistical Learning and Applications cs.LG The concept of median/consensus has been widely investigated in order to<br>provide a statistical summary of ranking data, i.e. realizations of a random<br>permutation $\Sigma$ of a finite set, $\{1,\; \ldots,\; n\}$ with $n\geq 1$<br>say. As it sheds light onto only one aspect of $\Sigma$'s distribution $P$, it<br>may neglect other informative features. It is the purpose of this paper to<br>define analogs of quantiles, ranks and statistical procedures based on such<br>quantities for the analysis of ranking data by means of a metric-based notion<br>of depth function on the symmetric group. Overcoming the absence of vector<br>space structure on $\mathfrak{S}_n$, the latter defines a center-outward<br>ordering of the permutations in the support of $P$ and extends the classic<br>metric-based formulation of consensus ranking (medians corresponding then to<br>the deepest permutations). The axiomatic properties ...</code>    | <code>Robust Consensus in Ranking Data Analysis: Definitions, Properties and<br>  Computational Issues cs.LG As the issue of robustness in AI systems becomes vital, statistical learning<br>techniques that are reliable even in presence of partly contaminated data have<br>to be developed. Preference data, in the form of (complete) rankings in the<br>simplest situations, are no exception and the demand for appropriate concepts<br>and tools is all the more pressing given that technologies fed by or producing<br>this type of data (e.g. search engines, recommending systems) are now massively<br>deployed. However, the lack of vector space structure for the set of rankings<br>(i.e. the symmetric group $\mathfrak{S}_n$) and the complex nature of<br>statistics considered in ranking data analysis make the formulation of<br>robustness objectives in this domain challenging. In this paper, we introduce<br>notions of robustness, together with dedicated statistical methods, for<br>Consensus Ranking the flagship problem in ranking data analysi...</code> | <code>Bootstrap your own latent: A new approach to self-supervised Learning cs.LG We introduce Bootstrap Your Own Latent (BYOL), a new approach to<br>self-supervised image representation learning. BYOL relies on two neural<br>networks, referred to as online and target networks, that interact and learn<br>from each other. From an augmented view of an image, we train the online<br>network to predict the target network representation of the same image under a<br>different augmented view. At the same time, we update the target network with a<br>slow-moving average of the online network. While state-of-the art methods rely<br>on negative pairs, BYOL achieves a new state of the art without them. BYOL<br>reaches $74.3\%$ top-1 classification accuracy on ImageNet using a linear<br>evaluation with a ResNet-50 architecture and $79.6\%$ with a larger ResNet. We<br>show that BYOL performs on par or better than the current state of the art on<br>both transfer and semi-supervised benchmarks. Our implementation and pretrained<br>models are g...</code> |
* Loss: [<code>TripletLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#tripletloss) with these parameters:
  ```json
  {
      "distance_metric": "TripletDistanceMetric.COSINE",
      "triplet_margin": 0.5
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 3
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 16
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step  | Training Loss |
|:------:|:-----:|:-------------:|
| 0.0898 | 500   | 0.1328        |
| 0.1796 | 1000  | 0.0642        |
| 0.2694 | 1500  | 0.0570        |
| 0.3593 | 2000  | 0.0509        |
| 0.4491 | 2500  | 0.0478        |
| 0.5389 | 3000  | 0.0473        |
| 0.6287 | 3500  | 0.0469        |
| 0.7185 | 4000  | 0.0442        |
| 0.8083 | 4500  | 0.0411        |
| 0.8981 | 5000  | 0.0384        |
| 0.9880 | 5500  | 0.0417        |
| 1.0778 | 6000  | 0.0338        |
| 1.1676 | 6500  | 0.0327        |
| 1.2574 | 7000  | 0.0301        |
| 1.3472 | 7500  | 0.0319        |
| 1.4370 | 8000  | 0.0316        |
| 1.5269 | 8500  | 0.0285        |
| 1.6167 | 9000  | 0.0311        |
| 1.7065 | 9500  | 0.0276        |
| 1.7963 | 10000 | 0.0284        |
| 1.8861 | 10500 | 0.0261        |
| 1.9759 | 11000 | 0.0268        |
| 2.0657 | 11500 | 0.0226        |
| 2.1556 | 12000 | 0.0232        |
| 2.2454 | 12500 | 0.0210        |
| 2.3352 | 13000 | 0.0233        |
| 2.4250 | 13500 | 0.0221        |
| 2.5148 | 14000 | 0.0205        |
| 2.6046 | 14500 | 0.0205        |
| 2.6944 | 15000 | 0.0209        |
| 2.7843 | 15500 | 0.0196        |
| 2.8741 | 16000 | 0.0213        |
| 2.9639 | 16500 | 0.0214        |


### Training Time
- **Training**: 1.2 hours

### Framework Versions
- Python: 3.13.1
- Sentence Transformers: 5.5.1
- Transformers: 5.12.1
- PyTorch: 2.12.0+cu126
- Accelerate: 1.14.0
- Datasets: 5.0.0
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### TripletLoss
```bibtex
@misc{hermans2017defense,
    title={In Defense of the Triplet Loss for Person Re-Identification},
    author={Alexander Hermans and Lucas Beyer and Bastian Leibe},
    year={2017},
    eprint={1703.07737},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->