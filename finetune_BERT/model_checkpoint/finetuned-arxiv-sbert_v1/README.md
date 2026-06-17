---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:11098
- loss:TripletLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: "Alzheimer's Dementia Recognition through Spontaneous Speech: The\
    \ ADReSS\n  Challenge eess.AS The ADReSS Challenge at INTERSPEECH 2020 defines\
    \ a shared task through which\ndifferent approaches to the automated recognition\
    \ of Alzheimer's dementia based\non spontaneous speech can be compared. ADReSS\
    \ provides researchers with a\nbenchmark speech dataset which has been acoustically\
    \ pre-processed and balanced\nin terms of age and gender, defining two cognitive\
    \ assessment tasks, namely:\nthe Alzheimer's speech classification task and the\
    \ neuropsychological score\nregression task. In the Alzheimer's speech classification\
    \ task, ADReSS\nchallenge participants create models for classifying speech as\
    \ dementia or\nhealthy control speech. In the the neuropsychological score regression\
    \ task,\nparticipants create models to predict mini-mental state examination scores.\n\
    This paper describes the ADReSS Challenge in detail and presents a baseline for\n\
    both tasks, including feature extraction procedures and results for\nclassification\
    \ and regression models. ADReSS aims to provide the speech and\nlanguage Alzheimer's\
    \ research community with a platform for comprehensive\nmethodological comparisons.\
    \ This will hopefully contribute to addressing the\nlack of standardisation that\
    \ currently affects the field and shed light on\navenues for future research and\
    \ clinical applicability."
  sentences:
  - 'Towards Understanding the Invertibility of Convolutional Neural Networks stat.ML
    Several recent works have empirically observed that Convolutional Neural Nets

    (CNNs) are (approximately) invertible. To understand this approximate

    invertibility phenomenon and how to leverage it more effectively, we focus on
    a

    theoretical explanation and develop a mathematical model of sparse signal

    recovery that is consistent with CNNs with random weights. We give an exact

    connection to a particular model of model-based compressive sensing (and its

    recovery algorithms) and random-weight CNNs. We show empirically that several

    learned networks are consistent with our mathematical analysis and then

    demonstrate that with such a simple theoretical framework, we can obtain

    reasonable re- construction results on real images. We also discuss gaps

    between our model assumptions and the CNN trained for classification in

    practical scenarios.'
  - 'eVAE: Evolutionary Variational Autoencoder cs.NE The surrogate loss of variational
    autoencoders (VAEs) poses various

    challenges to their training, inducing the imbalance between task fitting and

    representation inference. To avert this, the existing strategies for VAEs focus

    on adjusting the tradeoff by introducing hyperparameters, deriving a tighter

    bound under some mild assumptions, or decomposing the loss components per

    certain neural settings. VAEs still suffer from uncertain tradeoff learning.We

    propose a novel evolutionary variational autoencoder (eVAE) building on the

    variational information bottleneck (VIB) theory and integrative evolutionary

    neural learning. eVAE integrates a variational genetic algorithm into VAE with

    variational evolutionary operators including variational mutation, crossover,

    and evolution. Its inner-outer-joint training mechanism synergistically and

    dynamically generates and updates the uncertain tradeoff learning in the

    evidence lower bound (ELBO) without additional constraints. Apart from learning

    a lossy compression and representation of data under the VIB assumption, eVAE

    presents an evolutionary paradigm to tune critical factors of VAEs and deep

    neural networks and addresses the premature convergence and random search

    problem by integrating evolutionary optimization into deep learning.

    Experiments show that eVAE addresses the KL-vanishing problem for text

    generation with low reconstruction loss, generates all disentangled factors

    with sharp images, and improves the image generation quality,respectively. eVAE

    achieves better reconstruction loss, disentanglement, and generation-inference

    balance than its competitors.'
  - "The X-LANCE Technical Report for Interspeech 2024 Speech Processing\n  Using\
    \ Discrete Speech Unit Challenge eess.AS Discrete speech tokens have been more\
    \ and more popular in multiple speech\nprocessing fields, including automatic\
    \ speech recognition (ASR), text-to-speech\n(TTS) and singing voice synthesis\
    \ (SVS). In this paper, we describe the systems\ndeveloped by the SJTU X-LANCE\
    \ group for the TTS (acoustic + vocoder), SVS, and\nASR tracks in the Interspeech\
    \ 2024 Speech Processing Using Discrete Speech Unit\nChallenge. Notably, we achieved\
    \ 1st rank on the leaderboard in the TTS track\nboth with the whole training set\
    \ and only 1h training data, with the highest\nUTMOS score and lowest bitrate\
    \ among all submissions."
- source_sentence: 'Variational Mixture of Normalizing Flows stat.ML In the past few
    years, deep generative models, such as generative adversarial

    networks \autocite{GAN}, variational autoencoders \autocite{vaepaper}, and

    their variants, have seen wide adoption for the task of modelling complex data

    distributions. In spite of the outstanding sample quality achieved by those

    early methods, they model the target distributions \emph{implicitly}, in the

    sense that the probability density functions induced by them are not explicitly

    accessible. This fact renders those methods unfit for tasks that require, for

    example, scoring new instances of data with the learned distributions.

    Normalizing flows have overcome this limitation by leveraging the

    change-of-variables formula for probability density functions, and by using

    transformations designed to have tractable and cheaply computable Jacobians.

    Although flexible, this framework lacked (until recently

    \autocites{semisuplearning_nflows, RAD}) a way to introduce discrete structure

    (such as the one found in mixtures) in the models it allows to construct, in an

    unsupervised scenario. The present work overcomes this by using normalizing

    flows as components in a mixture model and devising an end-to-end training

    procedure for such a model. This procedure is based on variational inference,

    and uses a variational posterior parameterized by a neural network. As will

    become clear, this model naturally lends itself to (multimodal) density

    estimation, semi-supervised learning, and clustering. The proposed model is

    illustrated on two synthetic datasets, as well as on a real-world dataset.

    Keywords: Deep generative models, normalizing flows, variational inference,

    probabilistic modelling, mixture models.'
  sentences:
  - 'Reweighted Expectation Maximization stat.ML Training deep generative models with
    maximum likelihood remains a challenge.

    The typical workaround is to use variational inference (VI) and maximize a

    lower bound to the log marginal likelihood of the data. Variational

    auto-encoders (VAEs) adopt this approach. They further amortize the cost of

    inference by using a recognition network to parameterize the variational

    family. Amortized VI scales approximate posterior inference in deep generative

    models to large datasets. However it introduces an amortization gap and leads

    to approximate posteriors of reduced expressivity due to the problem known as

    posterior collapse. In this paper, we consider expectation maximization (EM) as

    a paradigm for fitting deep generative models. Unlike VI, EM directly maximizes

    the log marginal likelihood of the data. We rediscover the importance weighted

    auto-encoder (IWAE) as an instance of EM and propose a new EM-based algorithm

    for fitting deep generative models called reweighted expectation maximization

    (REM). REM learns better generative models than the IWAE by decoupling the

    learning dynamics of the generative model and the recognition network using a

    separate expressive proposal found by moment matching. We compared REM to the

    VAE and the IWAE on several density estimation benchmarks and found it leads to

    significantly better performance as measured by log-likelihood.'
  - 'Analyzing Multi-Task Learning for Abstractive Text Summarization cs.CL Despite
    the recent success of multi-task learning and pre-finetuning for

    natural language understanding, few works have studied the effects of task

    families on abstractive text summarization. Task families are a form of task

    grouping during the pre-finetuning stage to learn common skills, such as

    reading comprehension. To close this gap, we analyze the influence of

    multi-task learning strategies using task families for the English abstractive

    text summarization task. We group tasks into one of three strategies, i.e.,

    sequential, simultaneous, and continual multi-task learning, and evaluate

    trained models through two downstream tasks. We find that certain combinations

    of task families (e.g., advanced reading comprehension and natural language

    inference) positively impact downstream performance. Further, we find that

    choice and combinations of task families influence downstream performance more

    than the training scheme, supporting the use of task families for abstractive

    text summarization.'
  - "Variational Mixture of HyperGenerators for Learning Distributions Over\n  Functions\
    \ cs.LG Recent approaches build on implicit neural representations (INRs) to propose\n\
    generative models over function spaces. However, they are computationally\ncostly\
    \ when dealing with inference tasks, such as missing data imputation, or\ndirectly\
    \ cannot tackle them. In this work, we propose a novel deep generative\nmodel,\
    \ named VAMoH. VAMoH combines the capabilities of modeling continuous\nfunctions\
    \ using INRs and the inference capabilities of Variational Autoencoders\n(VAEs).\
    \ In addition, VAMoH relies on a normalizing flow to define the prior,\nand a\
    \ mixture of hypernetworks to parametrize the data log-likelihood. This\ngives\
    \ VAMoH a high expressive capability and interpretability. Through\nexperiments\
    \ on a diverse range of data types, such as images, voxels, and\nclimate data,\
    \ we show that VAMoH can effectively learn rich distributions over\ncontinuous\
    \ functions. Furthermore, it can perform inference-related tasks, such\nas conditional\
    \ super-resolution generation and in-painting, as well or better\nthan previous\
    \ approaches, while being less computationally demanding."
- source_sentence: 'Learning Gaussian Graphical Models with Observed or Latent FVSs
    cs.LG Gaussian Graphical Models (GGMs) or Gauss Markov random fields are widely

    used in many applications, and the trade-off between the modeling capacity and

    the efficiency of learning and inference has been an important research

    problem. In this paper, we study the family of GGMs with small feedback vertex

    sets (FVSs), where an FVS is a set of nodes whose removal breaks all the

    cycles. Exact inference such as computing the marginal distributions and the

    partition function has complexity $O(k^{2}n)$ using message-passing algorithms,

    where k is the size of the FVS, and n is the total number of nodes. We propose

    efficient structure learning algorithms for two cases: 1) All nodes are

    observed, which is useful in modeling social or flight networks where the FVS

    nodes often correspond to a small number of high-degree nodes, or hubs, while

    the rest of the networks is modeled by a tree. Regardless of the maximum

    degree, without knowing the full graph structure, we can exactly compute the

    maximum likelihood estimate in $O(kn^2+n^2\log n)$ if the FVS is known or in

    polynomial time if the FVS is unknown but has bounded size. 2) The FVS nodes

    are latent variables, where structure learning is equivalent to decomposing a

    inverse covariance matrix (exactly or approximately) into the sum of a

    tree-structured matrix and a low-rank matrix. By incorporating efficient

    inference into the learning steps, we can obtain a learning algorithm using

    alternating low-rank correction with complexity $O(kn^{2}+n^{2}\log n)$ per

    iteration. We also perform experiments using both synthetic data as well as

    real data of flight delays to demonstrate the modeling capacity with FVSs of

    various sizes.'
  sentences:
  - 'Fine-scale Surface Normal Estimation using a Single NIR Image cs.CV We present
    surface normal estimation using a single near infrared (NIR)

    image. We are focusing on fine-scale surface geometry captured with an

    uncalibrated light source. To tackle this ill-posed problem, we adopt a

    generative adversarial network which is effective in recovering a sharp output,

    which is also essential for fine-scale surface normal estimation. We

    incorporate angular error and integrability constraint into the objective

    function of the network to make estimated normals physically meaningful. We

    train and validate our network on a recent NIR dataset, and also evaluate the

    generality of our trained model by using new external datasets which are

    captured with a different camera under different environment.'
  - "Decentralized Learning of Tree-Structured Gaussian Graphical Models from\n  Noisy\
    \ Data cs.LG This paper studies the decentralized learning of tree-structured\
    \ Gaussian\ngraphical models (GGMs) from noisy data. In decentralized learning,\
    \ data set is\ndistributed across different machines (sensors), and GGMs are widely\
    \ used to\nmodel complex networks such as gene regulatory networks and social\
    \ networks.\nThe proposed decentralized learning uses the Chow-Liu algorithm for\
    \ estimating\nthe tree-structured GGM.\n  In previous works, upper bounds on the\
    \ probability of incorrect tree\nstructure recovery were given mostly without\
    \ any practical noise for\nsimplification. While this paper investigates the effects\
    \ of three common types\nof noisy channels: Gaussian, Erasure, and binary symmetric\
    \ channel. For\nGaussian channel case, to satisfy the failure probability upper\
    \ bound $\\delta >\n0$ in recovering a $d$-node tree structure, our proposed theorem\
    \ requires only\n$\\mathcal{O}(\\log(\\frac{d}{\\delta}))$ samples for the smallest\
    \ sample size\n($n$) comparing to the previous literature \\cite{Nikolakakis}\
    \ with\n$\\mathcal{O}(\\log^4(\\frac{d}{\\delta}))$ samples by using the positive\n\
    correlation coefficient assumption that is used in some important works in the\n\
    literature. Moreover, the approximately bounded Gaussian random variable\nassumption\
    \ does not appear in \\cite{Nikolakakis}. Given some knowledge about\nthe tree\
    \ structure, the proposed Algorithmic Bound will achieve obviously\nbetter performance\
    \ with small sample size (e.g., $< 2000$) comparing with\nformulaic bounds. Finally,\
    \ we validate our theoretical results by performing\nsimulations on synthetic\
    \ data sets."
  - "GripNet: Graph Information Propagation on Supergraph for Heterogeneous\n  Graphs\
    \ cs.LG Heterogeneous graph representation learning aims to learn low-dimensional\n\
    vector representations of different types of entities and relations to empower\n\
    downstream tasks. Existing methods either capture semantic relationships but\n\
    indirectly leverage node/edge attributes in a complex way, or leverage\nnode/edge\
    \ attributes directly without taking semantic relationships into\naccount. When\
    \ involving multiple convolution operations, they also have poor\nscalability.\
    \ To overcome these limitations, this paper proposes a flexible and\nefficient\
    \ Graph information propagation Network (GripNet) framework.\nSpecifically, we\
    \ introduce a new supergraph data structure consisting of\nsupervertices and superedges.\
    \ A supervertex is a semantically-coherent\nsubgraph. A superedge defines an information\
    \ propagation path between two\nsupervertices. GripNet learns new representations\
    \ for the supervertex of\ninterest by propagating information along the defined\
    \ path using multiple\nlayers. We construct multiple large-scale graphs and evaluate\
    \ GripNet against\ncompeting methods to show its superiority in link prediction,\
    \ node\nclassification, and data integration."
- source_sentence: 'AniPixel: Towards Animatable Pixel-Aligned Human Avatar cs.CV
    Although human reconstruction typically results in human-specific avatars,

    recent 3D scene reconstruction techniques utilizing pixel-aligned features show

    promise in generalizing to new scenes. Applying these techniques to human

    avatar reconstruction can result in a volumetric avatar with generalizability

    but limited animatability due to rendering only being possible for static

    representations. In this paper, we propose AniPixel, a novel animatable and

    generalizable human avatar reconstruction method that leverages pixel-aligned

    features for body geometry prediction and RGB color blending. Technically, to

    align the canonical space with the target space and the observation space, we

    propose a bidirectional neural skinning field based on skeleton-driven

    deformation to establish the target-to-canonical and canonical-to-observation

    correspondences. Then, we disentangle the canonical body geometry into a

    normalized neutral-sized body and a subject-specific residual for better

    generalizability. As the geometry and appearance are closely related, we

    introduce pixel-aligned features to facilitate the body geometry prediction and

    detailed surface normals to reinforce the RGB color blending. We also devise a

    pose-dependent and view direction-related shading module to represent the local

    illumination variance. Experiments show that AniPixel renders comparable novel

    views while delivering better novel pose animation results than

    state-of-the-art methods.'
  sentences:
  - "Vid2Avatar: 3D Avatar Reconstruction from Videos in the Wild via\n  Self-supervised\
    \ Scene Decomposition cs.CV We present Vid2Avatar, a method to learn human avatars\
    \ from monocular\nin-the-wild videos. Reconstructing humans that move naturally\
    \ from monocular\nin-the-wild videos is difficult. Solving it requires accurately\
    \ separating\nhumans from arbitrary backgrounds. Moreover, it requires reconstructing\n\
    detailed 3D surface from short video sequences, making it even more\nchallenging.\
    \ Despite these challenges, our method does not require any\ngroundtruth supervision\
    \ or priors extracted from large datasets of clothed\nhuman scans, nor do we rely\
    \ on any external segmentation modules. Instead, it\nsolves the tasks of scene\
    \ decomposition and surface reconstruction directly in\n3D by modeling both the\
    \ human and the background in the scene jointly,\nparameterized via two separate\
    \ neural fields. Specifically, we define a\ntemporally consistent human representation\
    \ in canonical space and formulate a\nglobal optimization over the background\
    \ model, the canonical human shape and\ntexture, and per-frame human pose parameters.\
    \ A coarse-to-fine sampling\nstrategy for volume rendering and novel objectives\
    \ are introduced for a clean\nseparation of dynamic human and static background,\
    \ yielding detailed and robust\n3D human geometry reconstructions. We evaluate\
    \ our methods on publicly\navailable datasets and show improvements over prior\
    \ art."
  - 'Rethinking Kernel Methods for Node Representation Learning on Graphs cs.LG Graph
    kernels are kernel methods measuring graph similarity and serve as a

    standard tool for graph classification. However, the use of kernel methods for

    node classification, which is a related problem to graph representation

    learning, is still ill-posed and the state-of-the-art methods are heavily based

    on heuristics. Here, we present a novel theoretical kernel-based framework for

    node classification that can bridge the gap between these two representation

    learning problems on graphs. Our approach is motivated by graph kernel

    methodology but extended to learn the node representations capturing the

    structural information in a graph. We theoretically show that our formulation

    is as powerful as any positive semidefinite kernels. To efficiently learn the

    kernel, we propose a novel mechanism for node feature aggregation and a

    data-driven similarity metric employed during the training phase. More

    importantly, our framework is flexible and complementary to other graph-based

    deep learning models, e.g., Graph Convolutional Networks (GCNs). We empirically

    evaluate our approach on a number of standard node classification benchmarks,

    and demonstrate that our model sets the new state of the art.'
  - "Automated Low-cost Terrestrial Laser Scanner for Measuring Diameters at\n  Breast\
    \ Height and Heights of Forest Trees cs.CV Terrestrial laser scanner is a kind\
    \ of fast, high-precision data acquisition\ndevice, which had been more and more\
    \ applied to the research areas of forest\ninventory. In this study, a kind of\
    \ automated low-cost terrestrial laser\nscanner was designed and implemented based\
    \ on a two-dimensional laser radar\nsensor SICK LMS-511 and a stepper motor. The\
    \ new scanner was named as BEE,\nwhich can scan the forest trees in three dimension.\
    \ The BEE scanner and its\nsupporting software are specifically designed for forest\
    \ inventory. The\nexperiments have been performed by using the BEE scanner in\
    \ an artificial\nginkgo forest which was located in Haidian district of Beijing.\
    \ Four square\nplots were selected to do the experiments. The BEE scanner scanned\
    \ in the four\nplots and acquired the single scan data respectively. The DBH,\
    \ tree height and\ntree position of trees in the four plots were estimated and\
    \ analyzed. For\ncomparison, the manual measured data was also collected in the\
    \ four plots. The\ntree stem detection rate for all four plots was 92.75%; the\
    \ root mean square\nerror of the DBH estimation was 1.27cm; the root mean square\
    \ error of the tree\nheight estimation was 0.24m; the tree position estimation\
    \ was in line with the\nactual position. Experimental results show that the BEE\
    \ scanner can efficiently\nestimate the structure parameters of forest trees and\
    \ has a good potential in\npractical application of forest inventory."
- source_sentence: 'Prompting Language Models for Linguistic Structure cs.CL Although
    pretrained language models (PLMs) can be prompted to perform a wide

    range of language tasks, it remains an open question how much this ability

    comes from generalizable linguistic understanding versus surface-level lexical

    patterns. To test this, we present a structured prompting approach for

    linguistic structured prediction tasks, allowing us to perform zero- and

    few-shot sequence tagging with autoregressive PLMs. We evaluate this approach

    on part-of-speech tagging, named entity recognition, and sentence chunking,

    demonstrating strong few-shot performance in all cases. We also find that while

    PLMs contain significant prior knowledge of task labels due to task leakage

    into the pretraining corpus, structured prompting can also retrieve linguistic

    structure with arbitrary labels. These findings indicate that the in-context

    learning ability and linguistic knowledge of PLMs generalizes beyond

    memorization of their training data.'
  sentences:
  - 'Scaling Instruction-Finetuned Language Models cs.LG Finetuning language models
    on a collection of datasets phrased as

    instructions has been shown to improve model performance and generalization to

    unseen tasks. In this paper we explore instruction finetuning with a particular

    focus on (1) scaling the number of tasks, (2) scaling the model size, and (3)

    finetuning on chain-of-thought data. We find that instruction finetuning with

    the above aspects dramatically improves performance on a variety of model

    classes (PaLM, T5, U-PaLM), prompting setups (zero-shot, few-shot, CoT), and

    evaluation benchmarks (MMLU, BBH, TyDiQA, MGSM, open-ended generation). For

    instance, Flan-PaLM 540B instruction-finetuned on 1.8K tasks outperforms PALM

    540B by a large margin (+9.4% on average). Flan-PaLM 540B achieves

    state-of-the-art performance on several benchmarks, such as 75.2% on five-shot

    MMLU. We also publicly release Flan-T5 checkpoints, which achieve strong

    few-shot performance even compared to much larger models, such as PaLM 62B.

    Overall, instruction finetuning is a general method for improving the

    performance and usability of pretrained language models.'
  - "Language Contamination Helps Explain the Cross-lingual Capabilities of\n  English\
    \ Pretrained Models cs.CL English pretrained language models, which make up the\
    \ backbone of many modern\nNLP systems, require huge amounts of unlabeled training\
    \ data. These models are\ngenerally presented as being trained only on English\
    \ text but have been found\nto transfer surprisingly well to other languages.\
    \ We investigate this\nphenomenon and find that common English pretraining corpora\
    \ actually contain\nsignificant amounts of non-English text: even when less than\
    \ 1% of data is not\nEnglish (well within the error rate of strong language classifiers),\
    \ this leads\nto hundreds of millions of foreign language tokens in large-scale\
    \ datasets. We\nthen demonstrate that even these small percentages of non-English\
    \ data\nfacilitate cross-lingual transfer for models trained on them, with target\n\
    language performance strongly correlated to the amount of in-language data seen\n\
    during pretraining. In light of these findings, we argue that no model is truly\n\
    monolingual when pretrained at scale, which should be considered when\nevaluating\
    \ cross-lingual transfer."
  - 'Intrinsic dimension of a dataset: what properties does one expect? cs.LG We propose
    an axiomatic approach to the concept of an intrinsic dimension of

    a dataset, based on a viewpoint of geometry of high-dimensional structures. Our

    first axiom postulates that high values of dimension be indicative of the

    presence of the curse of dimensionality (in a certain precise mathematical

    sense). The second axiom requires the dimension to depend smoothly on a

    distance between datasets (so that the dimension of a dataset and that of an

    approximating principal manifold would be close to each other). The third axiom

    is a normalization condition: the dimension of the Euclidean $n$-sphere $\s^n$

    is $\Theta(n)$. We give an example of a dimension function satisfying our

    axioms, even though it is in general computationally unfeasible, and discuss a

    computationally cheap function satisfying most but not all of our axioms (the

    ``intrinsic dimensionality'''' of Ch\''avez et al.)'
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
    'Prompting Language Models for Linguistic Structure cs.CL Although pretrained language models (PLMs) can be prompted to perform a wide\nrange of language tasks, it remains an open question how much this ability\ncomes from generalizable linguistic understanding versus surface-level lexical\npatterns. To test this, we present a structured prompting approach for\nlinguistic structured prediction tasks, allowing us to perform zero- and\nfew-shot sequence tagging with autoregressive PLMs. We evaluate this approach\non part-of-speech tagging, named entity recognition, and sentence chunking,\ndemonstrating strong few-shot performance in all cases. We also find that while\nPLMs contain significant prior knowledge of task labels due to task leakage\ninto the pretraining corpus, structured prompting can also retrieve linguistic\nstructure with arbitrary labels. These findings indicate that the in-context\nlearning ability and linguistic knowledge of PLMs generalizes beyond\nmemorization of their training data.',
    'Language Contamination Helps Explain the Cross-lingual Capabilities of\n  English Pretrained Models cs.CL English pretrained language models, which make up the backbone of many modern\nNLP systems, require huge amounts of unlabeled training data. These models are\ngenerally presented as being trained only on English text but have been found\nto transfer surprisingly well to other languages. We investigate this\nphenomenon and find that common English pretraining corpora actually contain\nsignificant amounts of non-English text: even when less than 1% of data is not\nEnglish (well within the error rate of strong language classifiers), this leads\nto hundreds of millions of foreign language tokens in large-scale datasets. We\nthen demonstrate that even these small percentages of non-English data\nfacilitate cross-lingual transfer for models trained on them, with target\nlanguage performance strongly correlated to the amount of in-language data seen\nduring pretraining. In light of these findings, we argue that no model is truly\nmonolingual when pretrained at scale, which should be considered when\nevaluating cross-lingual transfer.',
    'Scaling Instruction-Finetuned Language Models cs.LG Finetuning language models on a collection of datasets phrased as\ninstructions has been shown to improve model performance and generalization to\nunseen tasks. In this paper we explore instruction finetuning with a particular\nfocus on (1) scaling the number of tasks, (2) scaling the model size, and (3)\nfinetuning on chain-of-thought data. We find that instruction finetuning with\nthe above aspects dramatically improves performance on a variety of model\nclasses (PaLM, T5, U-PaLM), prompting setups (zero-shot, few-shot, CoT), and\nevaluation benchmarks (MMLU, BBH, TyDiQA, MGSM, open-ended generation). For\ninstance, Flan-PaLM 540B instruction-finetuned on 1.8K tasks outperforms PALM\n540B by a large margin (+9.4% on average). Flan-PaLM 540B achieves\nstate-of-the-art performance on several benchmarks, such as 75.2% on five-shot\nMMLU. We also publicly release Flan-T5 checkpoints, which achieve strong\nfew-shot performance even compared to much larger models, such as PaLM 62B.\nOverall, instruction finetuning is a general method for improving the\nperformance and usability of pretrained language models.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[ 1.0000,  0.7930, -0.1313],
#         [ 0.7930,  1.0000, -0.1298],
#         [-0.1313, -0.1298,  1.0000]])
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

* Size: 11,098 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>sentence_2</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence_0                                                                           | sentence_1                                                                           | sentence_2                                                                          |
  |:---------|:-------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type     | string                                                                               | string                                                                               | string                                                                              |
  | modality | text                                                                                 | text                                                                                 | text                                                                                |
  | details  | <ul><li>min: 63 tokens</li><li>mean: 220.62 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 64 tokens</li><li>mean: 210.02 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 52 tokens</li><li>mean: 208.1 tokens</li><li>max: 256 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | sentence_2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Divergent Ensemble Networks: Enhancing Uncertainty Estimation with<br>  Shared Representations and Independent Branching cs.LG Ensemble learning has proven effective in improving predictive performance<br>and estimating uncertainty in neural networks. However, conventional ensemble<br>methods often suffer from redundant parameter usage and computational<br>inefficiencies due to entirely independent network training. To address these<br>challenges, we propose the Divergent Ensemble Network (DEN), a novel<br>architecture that combines shared representation learning with independent<br>branching. DEN employs a shared input layer to capture common features across<br>all branches, followed by divergent, independently trainable layers that form<br>an ensemble. This shared-to-branching structure reduces parameter redundancy<br>while maintaining ensemble diversity, enabling efficient and scalable learning.</code>                                                                                                                                    | <code>Hydra: Preserving Ensemble Diversity for Model Distillation cs.LG Ensembles of models have been empirically shown to improve predictive<br>performance and to yield robust measures of uncertainty. However, they are<br>expensive in computation and memory. Therefore, recent research has focused on<br>distilling ensembles into a single compact model, reducing the computational<br>and memory burden of the ensemble while trying to preserve its predictive<br>behavior. Most existing distillation formulations summarize the ensemble by<br>capturing its average predictions. As a result, the diversity of the ensemble<br>predictions, stemming from each member, is lost. Thus, the distilled model<br>cannot provide a measure of uncertainty comparable to that of the original<br>ensemble. To retain more faithfully the diversity of the ensemble, we propose a<br>distillation method based on a single multi-headed neural network, which we<br>refer to as Hydra. The shared body network learns a joint feature<br>representation that enables each ...</code> | <code>Learning with Correntropy-induced Losses for Regression with Mixture of<br>  Symmetric Stable Noise cs.LG In recent years, correntropy and its applications in machine learning have<br>been drawing continuous attention owing to its merits in dealing with<br>non-Gaussian noise and outliers. However, theoretical understanding of<br>correntropy, especially in the statistical learning context, is still limited.<br>In this study, within the statistical learning framework, we investigate<br>correntropy based regression in the presence of non-Gaussian noise or outliers.<br>Motivated by the practical way of generating non-Gaussian noise or outliers, we<br>introduce mixture of symmetric stable noise, which include Gaussian noise,<br>Cauchy noise, and their mixture as special cases, to model non-Gaussian noise<br>or outliers. We demonstrate that under the mixture of symmetric stable noise<br>assumption, correntropy based regression can learn the conditional mean<br>function or the conditional median function well without resorting ...</code> |
  | <code>Improving Confidence in Evolutionary Mine Scheduling via Uncertainty<br>  Discounting cs.NE Mine planning is a complex task that involves many uncertainties. During<br>early stage feasibility, available mineral resources can only be estimated<br>based on limited sampling of ore grades from sparse drilling, leading to large<br>uncertainty in under-sampled parts of the deposit. Planning the extraction<br>schedule of ore over the life of a mine is crucial for its economic viability.<br>We introduce a new approach for determining an "optimal schedule under<br>uncertainty" that provides probabilistic bounds on the profits obtained in each<br>period. This treatment of uncertainty within an economic framework reduces<br>previously difficult-to-use models of variability into actionable insights. The<br>new method discounts profits based on uncertainty within an evolutionary<br>algorithm, sacrificing economic optimality of a single geological model for<br>improving the downside risk over an ensemble of equally likely models. We<br>p...</code> | <code>UBER: Uncertainty-Based Evolution with Large Language Models for<br>  Automatic Heuristic Design cs.NE NP-hard problem-solving traditionally relies on heuristics, but manually<br>crafting effective heuristics for complex problems remains challenging. While<br>recent work like FunSearch has demonstrated that large language models (LLMs)<br>can be leveraged for heuristic design in evolutionary algorithm (EA)<br>frameworks, their potential is not fully realized due to its deficiency in<br>exploitation and exploration. We present UBER (Uncertainty-Based Evolution for<br>Refinement), a method that enhances LLM+EA methods for automatic heuristic<br>design by integrating uncertainty on top of the FunSearch framework. UBER<br>introduces two key innovations: an Uncertainty-Inclusive Evolution Process<br>(UIEP) for adaptive exploration-exploitation balance, and a principled<br>Uncertainty-Inclusive Island Reset (UIIS) strategy for maintaining population<br>diversity. Through extensive experiments on challenging NP-complete problem...</code> | <code>Planning, Scheduling, and Uncertainty in the Sequence of Future Events cs.AI Scheduling in the factory setting is compounded by computational complexity<br>and temporal uncertainty. Together, these two factors guarantee that the<br>process of constructing an optimal schedule will be costly and the chances of<br>executing that schedule will be slight. Temporal uncertainty in the task<br>execution time can be offset by several methods: eliminate uncertainty by<br>careful engineering, restore certainty whenever it is lost, reduce the<br>uncertainty by using more accurate sensors, and quantify and circumscribe the<br>remaining uncertainty. Unfortunately, these methods focus exclusively on the<br>sources of uncertainty and fail to apply knowledge of the tasks which are to be<br>scheduled. A complete solution must adapt the schedule of activities to be<br>performed according to the evolving state of the production world. The example<br>of vision-directed assembly is presented to illustrate that the principle of<br>least commit...</code> |
  | <code>Ten years of image analysis and machine learning competitions in<br>  dementia cs.LG Machine learning methods exploiting multi-parametric biomarkers, especially<br>based on neuroimaging, have huge potential to improve early diagnosis of<br>dementia and to predict which individuals are at-risk of developing dementia.<br>To benchmark algorithms in the field of machine learning and neuroimaging in<br>dementia and assess their potential for use in clinical practice and clinical<br>trials, seven grand challenges have been organized in the last decade.<br>  The seven grand challenges addressed questions related to screening, clinical<br>status estimation, prediction and monitoring in (pre-clinical) dementia. There<br>was little overlap in clinical questions, tasks and performance metrics.<br>Whereas this aids providing insight on a broad range of questions, it also<br>limits the validation of results across challenges. The validation process<br>itself was mostly comparable between challenges, using similar methods for<br>ensuring...</code> | <code>Machine learning for the diagnosis of Parkinson's disease: A systematic<br>  review cs.LG Diagnosis of Parkinson's disease (PD) is commonly based on medical<br>observations and assessment of clinical signs, including the characterization<br>of a variety of motor symptoms. However, traditional diagnostic approaches may<br>suffer from subjectivity as they rely on the evaluation of movements that are<br>sometimes subtle to human eyes and therefore difficult to classify, leading to<br>possible misclassification. In the meantime, early non-motor symptoms of PD may<br>be mild and can be caused by many other conditions. Therefore, these symptoms<br>are often overlooked, making diagnosis of PD at an early stage challenging. To<br>address these difficulties and to refine the diagnosis and assessment<br>procedures of PD, machine learning methods have been implemented for the<br>classification of PD and healthy controls or patients with similar clinical<br>presentations (e.g., movement disorders or other Parkinsonian syndromes). To...</code> | <code>Optimally-Weighted Herding is Bayesian Quadrature cs.LG Herding and kernel herding are deterministic methods of choosing samples<br>which summarise a probability distribution. A related task is choosing samples<br>for estimating integrals using Bayesian quadrature. We show that the criterion<br>minimised when selecting samples in kernel herding is equivalent to the<br>posterior variance in Bayesian quadrature. We then show that sequential<br>Bayesian quadrature can be viewed as a weighted version of kernel herding which<br>achieves performance superior to any other weighted herding method. We<br>demonstrate empirically a rate of convergence faster than O(1/N). Our results<br>also imply an upper bound on the empirical error of the Bayesian quadrature<br>estimate.</code>                                                                                                                                                                                                                                                                            |
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
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.7205 | 500  | 0.1447        |
| 1.4409 | 1000 | 0.0662        |
| 2.1614 | 1500 | 0.0544        |
| 2.8818 | 2000 | 0.0439        |


### Training Time
- **Training**: 5.1 minutes

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