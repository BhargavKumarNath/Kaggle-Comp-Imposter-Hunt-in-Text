# Project Retrospective & Technical Write-Up: The Impostor Hunt Challenge

**Competition Link:** [Kaggle: Fake or Real – The Impostor Hunt](https://www.kaggle.com/competitions/fake-or-real-the-impostor-hunt)

This document details the end-to-end process of developing a machine learning model for the **"Impostor Hunt"** challenge. The objective was to perform **pairwise classification** on texts modified by Large Language Models (LLMs), identifying the "real" (less modified) text from the "fake" (more modified) one.  

Our iterative approach began with a strong baseline using **Gradient Boosted Decision Trees (GBDTs)** on engineered features, achieving a high CV score of **0.873**. Multiple attempts to fine-tune advanced transformer architectures failed due to the **extremely small dataset size (N=93)**.  

The project's major breakthrough came from pivoting to a **feature-extraction paradigm**. By combining our initial structural features with **deep semantic features** extracted from a sentence-transformer model, we created a **"Super Feature" set**. A final **XGBoost model** trained on these features achieved an outstanding **cross-validation accuracy of 0.926**, which became our final, most robust solution.  

This project demonstrates how, on small datasets, **sophisticated feature engineering** can decisively outperform end-to-end deep learning.

---

## 1. Problem Understanding and Framing

The core task was to distinguish between two texts in a pair: one "real" and one "fake." Both texts originated from the same source but were significantly modified by LLMs. The evaluation metric was **pairwise accuracy**, measuring the model's ability to correctly identify the "real" text for each pair.

### Key Constraint
The extremely small dataset size (~93 training samples) shaped the entire strategy. It flagged the **risk of overfitting** and **instability of complex deep learning models**.

### Initial Insights from EDA
Our Exploratory Data Analysis revealed statistically significant differences between real and fake texts, particularly in length:

- **Word Count:** Fake texts were, on average, twice as long as real texts (mean: 428 vs. 216).  
- **Sentence Count:** Fake texts contained more than three times as many sentences (mean: 37.8 vs. 11.7).  

**Hypothesis:** Structural and stylistic features, especially those related to verbosity, would be highly predictive.

---

## 2. Approach, Planning, and Iteration

We adopted a **structured, iterative development plan** to progressively build complexity and adapt based on empirical results.

### Phase 1: Strong Baseline
- Goal: Create a **fast, interpretable, and powerful baseline**.  
- Approach: Engineer a rich set of structural and stylistic features and train **GBDT models** (LightGBM, XGBoost).  
- Outcome: Established a **benchmark to beat**.

### Phase 2: Advanced Deep Learning
- Goal: Leverage transformers for nuanced text understanding.  
- Approach: Fine-tune a **Cross-Encoder model (DeBERTa-v3)**, standard for pairwise text classification.  
- Challenge: Dataset size was too small, leading to unstable fine-tuning results.

### Phase 3: Refinement and Ensembling
- Goal: Combine predictions from the best GBDT and transformer models.  
- Approach: Explore additional features and finalize a **submission pipeline**.  

**Note:** The plan was flexible; results from each phase informed the strategy for the next, leading to significant pivots from the original roadmap.

## 3. Implemented Solutions and Methods

Our development journey involved several distinct methods, with the successes and failures of each providing valuable lessons.

### A. The Champion Model: Feature Engineering + XGBoost

This approach proved to be the **most effective** and formed the core of our final solution.

#### Structural Feature Engineering
We created a feature set capturing the high-level characteristics of each text, including:

- **Basic Counts:** Character, word, and sentence counts.  
- **Readability Scores:** Flesch Reading Ease to measure textual complexity.  
- **Stylistic Ratios:** Average word/sentence length.  

#### Pairwise Feature Creation
The most critical step was creating features that explicitly compared `text_1` and `text_2`. For every base feature (e.g., `word_count`), we created `_diff` and `_ratio` features. This allowed the model to learn relative rules like:

> "The text with a much higher word count is likely fake."

#### Semantic Feature Extraction (The Breakthrough)
After hitting a wall with fine-tuning, we pivoted to **using a pre-trained `sentence-transformers/all-mpnet-base-v2` model** for feature extraction. We did not fine-tune this model. Instead, we generated **768-dimensional embeddings** for each text and created powerful semantic features:

- **Cosine Similarity:** Measures semantic similarity between the two texts.  
- **Element-wise Difference:** A 768-dimensional vector representing the semantic gap between the texts.  

#### Modeling
The combination of structural and semantic features (the **"Super Feature" set**) was used to train an **XGBoost classifier**, with hyperparameters optimized using **Optuna**.

### B. The Dead Ends: Transformer Fine-Tuning Attempts

These experiments were vital, as their failure proved the **superiority of the feature-based approach** for this specific problem.

- **Cross-Encoder:** Fine-tuning a `microsoft/deberta-v3-small` model completely failed. Validation loss remained ~0.693 (random guessing), and accuracy hovered around 50%.  
- **Siamese Difference Network:** A custom shared-encoder model with an explicit difference head also failed to learn, confirming that end-to-end fine-tuning was unviable on this dataset.

---

## 4. Results Obtained

| Model / Method                           | CV Pairwise Accuracy | Key Insight                                                                 |
|------------------------------------------|-------------------|----------------------------------------------------------------------------|
| XGBoost on Structural Features           | 0.873             | Strong baseline; structural cues are highly predictive.                   |
| XGBoost on "Super Features" (Structural + Semantic) | 0.926             | Breakthrough; semantic features provided a massive boost.                 |
| LightGBM + XGBoost Ensemble on "Super Features" | 0.926             | No improvement; indicates feature strength saturation.                    |
| All Fine-Tuning Attempts (Cross-Encoder, Siamese) | ~0.50 - 0.58      | Fine-tuning not viable for this dataset.                                   |
| Final Public Leaderboard Score           | 0.853             | Good generalization of the best model.                                     |

---

## 5. Challenges and Resolutions

### Challenge 1: Transformer Fine-Tuning Failure
- **Problem:** Standard transformer fine tuning failed completely due to the small dataset.  
- **Resolution:** Pivoted to **feature extraction** using pre-trained embeddings rather than end to end learning. This was the single most important decision.

### Challenge 2: Performance Plateau
- **Problem:** After achieving CV 0.926, adding advanced features like GPT-2 perplexity or ensembling yielded no gains.  
- **Resolution:** Interpreted as **signal saturation**; avoided further complexity to prevent overfitting.

---

## 6. Path to Final Submission

The final submission was a **direct operationalization** of our best-performing model:

1. **Consolidate the Pipeline:** Created a clean pipeline for structural features, pairwise comparisons, and semantic features via the sentence-transformer.  
2. **Train on Full Data:** XGBoost with Optuna-tuned hyperparameters trained on **100% of training data**.  
3. **Predict and Format:** Applied the pipeline to test data to generate the "Super Feature" set and formatted predictions into the required `id,real_text_id` CSV format.  
