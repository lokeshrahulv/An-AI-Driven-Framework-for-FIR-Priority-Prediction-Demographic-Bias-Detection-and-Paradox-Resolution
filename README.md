# An-AI-Driven-Framework-for-FIR-Priority-Prediction-Demographic-Bias-Detection-and-Paradox-Resolution

# An AI-Driven Framework for FIR Priority Prediction, Demographic Bias Detection, and Paradox Resolution

> **P. Sundaravadivel¹ · Lokesh Rahul V V² · Raja R³**
>
> ¹ Department of Artificial Intelligence and Machine Learning, Saveetha Engineering College, Chennai, India
> ² ³ Department of Computer Science Engineering (Cyber Security), Saveetha Engineering College, Chennai, India
>
> 📧 Corresponding author: [sundaravadivelp@saveetha.ac.in](mailto:sundaravadivelp@saveetha.ac.in)

---

## 📌 Abstract

The **First Information Report (FIR)** is the foundational document of the Indian criminal justice system, yet its processing remains prone to delays, inconsistent prioritization, demographic bias, and contradictory complaint records. Although initiatives such as the Crime and Criminal Tracking Network and Systems (CCTNS) have enabled digitization, intelligent decision-support mechanisms are largely absent.

This paper proposes an **AI-driven FIR management framework** integrating three components:

- **(i)** A multi-class **priority prediction module** using TF-IDF features with machine learning classifiers achieving a weighted **F1-score of 80.16%**
- **(ii)** A **demographic bias detection module** using the Disparate Impact Ratio to evaluate fairness across gender and age groups
- **(iii)** A novel **FIR Paradox resolution engine** that identifies semantically similar complaints assigned conflicting priority labels using cosine similarity

Experimental evaluation on **10,000 FIR records** demonstrates improved prioritization consistency, fairness monitoring, and transparency.

**Keywords:** `FIR automation` `priority prediction` `NLP` `TF-IDF` `random forest` `demographic bias detection` `FIR paradox` `cosine similarity` `criminal justice systems` `e-governance` `machine learning`

---

## 📋 Table of Contents

1. [Introduction](#1-introduction)
2. [Related Work](#2-related-work)
3. [Problem Formulation](#3-problem-formulation)
4. [Proposed System Architecture](#4-proposed-system-architecture)
5. [NLP-Based Priority Prediction Module](#5-nlp-based-priority-prediction-module)
6. [Demographic Bias Detection Module](#6-demographic-bias-detection-module)
7. [FIR Paradox Resolution Module](#7-fir-paradox-resolution-module)
8. [Experimental Evaluation](#8-experimental-evaluation)
9. [Discussion](#9-discussion)
10. [Conclusion](#10-conclusion)
11. [Declarations](#11-declarations)
12. [References](#12-references)

---

## 1. Introduction

The First Information Report (FIR) serves as the primary legal document for recording cognizable offenses in India under the Code of Criminal Procedure (CrPC), 1973. Despite nationwide digitization efforts, FIR management remains susceptible to operational inefficiencies such as:

- Delayed registration
- Inconsistent priority assignment
- Demographic bias
- Undetected duplicate or contradictory complaints

The **Crime and Criminal Tracking Network and Systems (CCTNS)** initiative has strengthened digital infrastructure; however, it primarily supports storage and retrieval functions without providing intelligent decision-support mechanisms. Consequently, case prioritization often depends on subjective human judgment, increasing the risk of inconsistency and potential demographic disparity.

Advances in **Natural Language Processing (NLP)** and **Machine Learning (ML)** have shown promising results in legal text classification and decision prediction. However, their integrated application within FIR workflows remains limited. Existing systems do not simultaneously address automated priority prediction, fairness auditing, and contradiction detection.

This paper proposes an AI-driven FIR management framework integrating:

| Component | Description |
|-----------|-------------|
| Priority Prediction | Multi-class classification using TF-IDF and ML classifiers |
| Bias Detection | Demographic fairness auditing via Disparate Impact Ratio |
| Paradox Resolution | Detection of semantically similar FIRs with conflicting labels |

---

## 2. Related Work

### Digital Transformation in Policing
The CCTNS initiative improved digitization of FIR records, enabling centralized storage and inter-departmental coordination. However, existing implementations primarily focus on **record management** rather than predictive analytics or intelligent decision support.

### Legal Text Classification
Classical ML methods such as **Support Vector Machines** have demonstrated strong performance in structured legal corpora, while recent transformer-based models have further enhanced classification accuracy. These approaches, however, are primarily designed for formal court judgments and do not directly address short, semi-structured FIR complaint narratives.

### Fairness-Aware Machine Learning
Fairness-aware ML has gained prominence due to concerns about demographic bias in predictive systems, with metrics such as the **Disparate Impact Ratio** used to quantify disparity across sensitive groups.

### Document Similarity
TF-IDF and cosine similarity remain effective for near-duplicate detection in domain-specific corpora. However, the detection of semantically similar FIR complaints assigned **conflicting priority labels** has not been formally addressed.

> **⚠️ Research Gap:** Existing systems focus on digitization or isolated predictive tasks but do not integrate priority prediction, fairness auditing, and contradiction detection within a unified FIR management framework.

---

## 3. Problem Formulation

### 3.1 FIR Priority Classification

Let $\mathcal{D} = \{(x_i, d_i, y_i)\}_{i=1}^{N}$ denote a dataset of $N$ FIR records, where:
- $x_i$ = complaint text
- $d_i \in \mathbb{R}^k$ = demographic features (age, gender)
- $y_i \in \mathcal{Y} = \{\text{High, Medium, Low}\}$ = priority label

The classification function:

$$f: \mathcal{X} \times \mathbb{R}^k \rightarrow \mathcal{Y}$$

minimises the weighted loss:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c \in \mathcal{Y}} w_c \cdot \mathbf{1}[y_i = c] \cdot \log P(\hat{y}_i = c \mid x_i, d_i)$$

**TF-IDF** feature extraction is defined as:

$$\text{TF-IDF}(t, x_i) = \text{tf}(t, x_i) \cdot \log\left(\frac{N}{1 + \text{df}(t)}\right)$$

The final feature representation concatenates textual and demographic features:

$$\Phi_i = [\phi(x_i) \parallel d_i]$$

### 3.2 Demographic Bias Quantification

For each demographic group $g$ and priority class $c$, the priority assignment rate is:

$$R(c \mid g) = \frac{|\{i : y_i = c \wedge s_i \in g\}|}{|\{i : s_i \in g\}|}$$

The **Disparate Impact Ratio (DIR)** between groups $g_j$ and $g_k$:

$$\text{DIR}(c, g_j, g_k) = \frac{R(c \mid g_j)}{R(c \mid g_k)}$$

Under the **four-fifths rule**, fairness is satisfied when:

$$0.80 \leq \text{DIR} \leq 1.25$$

### 3.3 Definition of the FIR Paradox

An **FIR Paradox** occurs when two complaint records are semantically similar yet assigned different priority labels. Formally, a paradox exists if:

$$\text{sim}(x_i, x_j) \geq \theta \quad \wedge \quad y_i \neq y_j$$

where cosine similarity is:

$$\text{sim}(x_i, x_j) = \frac{\phi(x_i) \cdot \phi(x_j)}{\|\phi(x_i)\| \, \|\phi(x_j)\|}$$

and $\theta = 0.65$ (empirically determined).

---

## 4. Proposed System Architecture

The framework follows a **modular three-tier architecture**:

![System Architecture](fig_architecture.png)
*Fig. 1 — Three-tier architecture of the proposed AI-driven FIR management framework consisting of presentation, application, and data layers.*

### 4.1 Overall System Design

```
FIR Submission
     │
     ▼
Text Preprocessing & TF-IDF Feature Extraction
     │
     ▼
Priority Classification (RF / LR / SVM)
     │
     ├──► Bias Detection Module (DIR)
     │
     ├──► FIR Paradox Engine (Cosine Similarity)
     │
     ▼
Database Storage + Full Audit Trail
```

### 4.2 Presentation Layer

| Portal | Users | Functionality |
|--------|-------|---------------|
| Citizen Portal | Public | FIR submission & status tracking |
| Police Dashboard | Officers | Priority levels, paradox alerts, fairness reports |
| Judiciary Interface | Judiciary | Read-only access to approved case records |

### 4.3 Application Layer

Core intelligence modules operating **sequentially** upon FIR submission:
- Text preprocessing
- TF-IDF feature extraction
- Priority prediction
- Demographic bias computation (DIR)
- Paradox detection (cosine similarity)

### 4.4 Data Layer

Document-oriented database storing:
- FIR records & demographic attributes
- Predicted priority labels
- Similarity flags
- Officer verification logs *(all audit-logged)*

### 4.5 Role-Based Access Control (RBAC)

| Role | Permissions |
|------|-------------|
| Citizen | Submit FIR, view own case status |
| Officer | View predictions, resolve paradoxes, update records |
| Judiciary | Read-only access to approved records |

---

## 5. NLP-Based Priority Prediction Module

### 5.1 Data Preprocessing Pipeline

```
Raw FIR Text
    │
    ├─ Lowercasing
    ├─ Remove non-alphanumeric characters
    ├─ Stopword elimination
    └─ Porter stemming
         │
         ▼
    Cleaned Text + Encoded Demographics
```

### 5.2 Feature Extraction Using TF-IDF

- **Representation:** Unigram + bigram
- **Vocabulary size:** 500 features
- **Final matrix:** Sparse TF-IDF vectors concatenated with demographic attributes

### 5.3 Classification Models

| Model | Strength |
|-------|----------|
| Random Forest | Interpretability via feature importance |
| Logistic Regression | High-dimensional sparse feature spaces |
| Linear SVM | High-dimensional sparse feature spaces |

**Train/Test split:** Stratified 80/20

### 5.4 Feature Importance Analysis

![Feature Importance](fig3_feature_importance.png)
*Fig. 2 — Top 20 feature importance scores from the Random Forest classifier.*

Top influential features include crime-specific keywords:
- `murder` · `kidnap` · `robbery` → **High priority**
- `noise complaint` · `lost wallet` · `minor argument` → **Low priority**
- `age` and `sex` also contribute measurable predictive signals

---

## 6. Demographic Bias Detection Module

### 6.1 Sensitive Attributes

| Attribute | Groups |
|-----------|--------|
| Gender | Male, Female |
| Age | Youth (18–25), Adult (26–40), Middle Age (41–60), Senior (61–70) |

### 6.2 Gender Bias Analysis

DIR values for all priority levels remain within the acceptable fairness range — the classification model does **not** introduce measurable gender-based disparity.

### 6.3 Age Group Bias Analysis

All DIR values remain within fairness thresholds across all four age cohorts, suggesting **demographic parity** across age categories.

### 6.4 Bias Detection Algorithm

```
For each demographic group g:
    Compute R(c | g) for each priority class c
    For each pair (gⱼ, gₖ):
        Compute DIR(c, gⱼ, gₖ)
        If DIR < 0.80 or DIR > 1.25:
            Flag for administrative review
```

---

## 7. FIR Paradox Resolution Module

### 7.1 Semantic Similarity Computation

Each FIR complaint → TF-IDF vector → pairwise cosine similarity computed between all complaint vectors.

### 7.2 Threshold Selection

| Similarity Range | Interpretation |
|-----------------|----------------|
| < 0.40 | Unrelated complaints |
| 0.40 – 0.65 | Ambiguous zone |
| ≥ 0.65 | Structurally similar narratives — paradox candidates |

**Operational threshold:** $\theta = 0.65$

### 7.3 Paradox Detection Algorithm

```python
paradox_set = []
for i in range(len(FIR_records)):
    for j in range(i+1, len(FIR_records)):
        sim = cosine_similarity(phi(x_i), phi(x_j))
        if sim >= theta and y_i != y_j:
            paradox_set.append((i, j, sim))
            flag_in_database(i, j)
```

### 7.4 Paradox Resolution Workflow

```
Paradox Detected
      │
      ▼
Both records flagged with ⚠️ Conflict Alert
      │
      ▼
Police Dashboard: side-by-side comparison
  ├─ Complaint texts
  ├─ Similarity score
  └─ Assigned priority labels
      │
      ▼
Officer Action:
  ├─ Confirm one label
  ├─ Modify classifications
  └─ Escalate for senior review
      │
      ▼
All actions → Audit Log
```

### 7.5 Scalability

Pairwise similarity has **O(n²)** complexity. For large deployments:
- Jurisdiction-level partitioning
- Approximate nearest neighbour (ANN) search

---

## 8. Experimental Evaluation

### 8.1 Dataset Description

| Split | Records |
|-------|---------|
| Training | 8,000 (80%) |
| Test | 2,000 (20%) |
| **Total** | **10,000** |

**Priority Class Distribution:**

| Priority | Count | Percentage |
|----------|-------|------------|
| High | 4,471 | 44.71% |
| Medium | 2,719 | 27.19% |
| Low | 2,810 | 28.10% |

**Dataset spans 12 crime categories.**

### 8.2 Experimental Setup

- **Language:** Python
- **Library:** scikit-learn
- **TF-IDF:** max 500 features, unigram + bigram
- **Metrics:** Accuracy, weighted F1-score, Precision, Recall, Confusion Matrix, ROC-AUC

### 8.3 Priority Prediction Performance

![Model Comparison](fig5_model_comparison.png)
*Fig. 3 — Performance comparison across evaluated classifiers.*

| Model | Accuracy | F1 | Precision | Recall |
|-------|----------|----|-----------|--------|
| Random Forest | 0.7270 | 0.7265 | 0.7261 | 0.7270 |
| Logistic Regression | 0.7955 | **0.8016** | **0.8147** | 0.7955 |
| Linear SVM | 0.7955 | **0.8016** | **0.8147** | 0.7955 |

> ✅ **Best weighted F1-score: 80.16%** (Logistic Regression & Linear SVM)

### 8.4 Confusion Matrix and ROC Analysis

![ROC Curves](fig2_roc_curves.png)
*Fig. 4 — ROC curves for multi-class priority classification under one-vs-rest evaluation.*

![Confusion Matrices](fig1_confusion_matrices.png)
*Fig. 5 — Confusion matrices for the evaluated classifiers on the test dataset.*

- Most misclassifications occur between **Medium** and **High** priority classes (semantic overlap)
- Highest AUC observed for **High-priority** classification

### 8.5 Bias Detection Results

![Bias Analysis](fig4_bias_analysis.png)
*Fig. 6 — Priority distribution across demographic groups confirming absence of systematic bias.*

**Disparate Impact Ratio (DIR) values — all within [0.80, 1.25] ✅**

| Attribute | High | Medium | Low |
|-----------|------|--------|-----|
| Gender (F vs M) | 1.008 | 0.957 | 1.047 |
| Youth vs Adult | 1.021 | 0.912 | 1.063 |
| Youth vs Middle Age | 1.035 | 0.894 | 1.082 |
| Youth vs Senior | 1.048 | 0.875 | 1.103 |
| Adult vs Senior | 1.026 | 0.961 | 1.038 |

> ✅ No measurable demographic disparity detected across any group.

### 8.6 Paradox Detection Results

![Paradox Detection](fig6_paradox_detection.png)
*Fig. 7 — Cosine similarity distribution showing 26,674 candidate pairs in the high-similarity zone (sim ≥ 0.65), of which 8,415 are confirmed paradox pairs with conflicting priority labels, at threshold θ = 0.65.*

| Metric | Value |
|--------|-------|
| Stratified sample size | 800 records |
| Candidate pairs (sim ≥ 0.65) | 26,674 |
| **Confirmed paradox pairs** | **8,415** |

**Paradox breakdown by conflict type:**

| Conflict Type | Pairs |
|---------------|-------|
| High ↔ Medium | 4,124 |
| Medium ↔ Low | 3,378 |
| High ↔ Low | 913 |
| **Total** | **8,415** |

### 8.7 Comparative Analysis

| Feature | Traditional Systems | Proposed Framework |
|---------|--------------------|--------------------|
| Digitization | ✅ | ✅ |
| Priority Prediction | ❌ | ✅ |
| Fairness Auditing | ❌ | ✅ |
| Paradox Detection | ❌ | ✅ |
| Explainability | ❌ | ✅ |
| Audit Trail | Partial | ✅ Full |

---

## 9. Discussion

The experimental results demonstrate that the proposed framework effectively integrates automated priority classification, demographic fairness auditing, and contradiction detection within a unified FIR management pipeline.

**Key findings:**

- TF-IDF representations combined with **linear classifiers remain effective** for structured complaint narratives
- Priority assignments show **no measurable demographic disparity** across gender and age groups
- A **substantial number of paradoxes** (8,415 confirmed pairs) exist in practice, highlighting inherent ambiguity in complaint narratives

**Limitations:**
- The structured dataset may not fully capture **linguistic diversity** and regional variation in real-world FIR submissions
- TF-IDF does not capture **deep contextual semantics**, potentially missing paraphrased but equivalent complaints

**Future Work:**
- Integration of **contextual embeddings** (BERT, RoBERTa)
- Large-scale deployment validation on real-world CCTNS data
- Extension to **regional languages**

---

## 10. Conclusion

This paper presented an AI-driven framework for intelligent FIR management integrating priority prediction, demographic bias detection, and paradox resolution within a unified decision-support pipeline.

| Achievement | Result |
|-------------|--------|
| Weighted F1-score | **80.16%** (Logistic Regression & Linear SVM) |
| Bias detection | All DIR values within [0.80, 1.25] ✅ |
| Paradox pairs detected | **8,415** confirmed conflicts |

By combining predictive analytics, fairness auditing, and contradiction detection, the framework advances beyond traditional digitization systems and enhances **transparency, accountability, and operational reliability** in FIR processing.

---

## 11. Declarations

| Item | Statement |
|------|-----------|
| **Funding** | No specific funding received |
| **Conflict of Interest** | None declared |
| **Ethics Approval** | Not required — no human participants, no PII in dataset |
| **Data Availability** | Available from corresponding author upon reasonable request |
| **Code Availability** | Available from corresponding author upon reasonable request |

### Author Contributions

- **P. Sundaravadivel** — Conceptualization, supervision
- **Lokesh Rahul V V** — System architecture, ML implementation, experiments
- **Raja R** — System development, evaluation, manuscript preparation

---

## 12. References

1. Ministry of Home Affairs, Government of India. *Crime and Criminal Tracking Network and Systems (CCTNS): Project Overview and Implementation Status.* National Crime Records Bureau, New Delhi, India (2020)

2. Aletras, N., Tsarapatsanis, D., Preotiuc-Pietro, D., Lampos, V. Predicting judicial decisions of the European Court of Human Rights: A natural language processing perspective. *PeerJ Computer Science* **2**, 93 (2016)

3. Luo, B., Feng, F., Xu, J., Zhang, X., Zhao, D. Learning to predict charges for criminal cases with legal basis. In: *Proceedings of the Conference on Empirical Methods in Natural Language Processing*, pp. 2727–2736 (2017)

4. Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., Galstyan, A. A survey on bias and fairness in machine learning. *ACM Computing Surveys* **54**(6), 1–35 (2021)

---

<div align="center">

**Saveetha Engineering College, Chennai, India**

*Submitted to Springer Nature*

</div>
