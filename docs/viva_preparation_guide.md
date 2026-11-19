# 🎓 SmartHire: Master Viva & Presentation Guide

> **Your 2-Day Survival Plan**: This guide is designed so that even if you started with zero knowledge, you and your teammate can confidently present this project, demonstrate the working app, and answer any question the external examiner asks.

---

## 👥 Team Work Division (2 People)

| Dimension | **Person A (Modeling & Generation Lead)** | **Person B (Evaluation & Prompt Engineering Lead)** |
| :--- | :--- | :--- |
| **Project Role** | Machine Learning & Decoding Specialist | NLP Metrics, Prompts & UI Architect |
| **Core Modules** | `train_evaluator.py`, `evaluator.py`, `generator.py` | `metrics.py`, `prompts.py`, `app.py` |
| **Syllabus Units** | Unit 2 (LLMs & Fine-Tuning), Lab 1 & 2, Lab 3 & 4 | Unit 3 (Text Generation Metrics), Unit 5 (Prompting) |
| **Live Demo Duties** | Opens **Tab 4 (Training Results)** and **Tab 2 (Decoding Lab)**; explains loss curves and sampling sliders. | Opens **Tab 1 (Interview Session)** and **Tab 3 (Evaluation Metrics)**; runs candidate response and explains BLEU/ROUGE. |

---

## 🎤 2-Minute Presentation Script (Say this at the start!)

### Person A begins:
> *"Good morning, Sir/Ma'am. For our Generative AI course project, we built **SmartHire: an AI-Powered Mock Interviewer and Answer Evaluation Coach**. 
> Rather than relying on third-party commercial APIs, we engineered an end-to-end local Generative AI pipeline using PyTorch and Hugging Face Transformers. 
> On the generation side, we implemented a custom autoregressive language model where we wrote the token decoding algorithms—Greedy, Temperature, Top-K, and Top-P Nucleus sampling—from scratch to control how interview questions are dynamically generated."*

### Person B continues:
> *"On the evaluation side, we fine-tuned a DistilBERT Transformer classifier on a custom technical interview dataset using cross-entropy loss and AdamW optimization, achieving **94.4% validation accuracy** across four competency tiers. 
> To evaluate candidate answers against benchmark answers, our system calculates precision-based BLEU scores and recall-based ROUGE scores in real-time, coupled with Chain-of-Thought reasoning to deliver constructive candidate feedback. We will now demonstrate the system live."*

---

## 🖥️ Live Demo Walkthrough (How to show the app)

To launch the app on your computer:
```bash
cd SmartHire
python -m streamlit run app.py
```
It will open in your browser at `http://localhost:8501`.

### 1. Show Tab 4: "Training Results" (Person A presents)
- Point to the **Training Loss Curve**: Explain that initial loss was $1.38$ and dropped smoothly to $0.62$ over 5 epochs.
- Point to the **Classification Report**: Highlight the **$94.44\%$ validation accuracy** across 4 classes (Poor, Needs Improvement, Good, Excellent).
- **Viva Hook**: Explain that we used `[CLS]` token pooling and AdamW weight decay ($0.01$).

### 2. Show Tab 2: "Decoding Lab" (Person A presents)
- Enter a prompt like *"In Python, the primary difference between a list and a tuple is"*.
- Click **"Compare All Strategies"**:
  - Show **Greedy Search**: Explain that it deterministically picks the highest probability word every step.
  - Show **Temperature**: Explain that higher temperature makes the distribution flatter and more creative.
  - Show **Top-P (Nucleus)**: Point out how it dynamically adapts the candidate pool based on cumulative probability mass.
- Click **"Generate"** to show the **Token-by-Token Probability Table**: point to the top-5 candidate words and their exact probability values at each step!

### 3. Show Tab 1: "Interview Session" (Person B presents)
- Select **Role**: *Python Developer*, **Difficulty**: *Medium*.
- Click **Start Interview**.
- Type a strong technical answer or a brief one.
- Show the instant **Score Card**, **Confidence Rating**, and **Chain-of-Thought Feedback**:
  - *"Notice how the system breaks down evaluation into Relevance Check, Concept Coverage, Technical Depth, and Communication Quality before assigning a score."*

### 4. Show Tab 3: "Evaluation Metrics" (Person B presents)
- Compare a reference answer with a student answer.
- Point out **BLEU** (measures Precision: how many generated words appear in the reference) vs **ROUGE** (measures Recall: how many reference key points the candidate captured).

---

## 🎯 Top 20 Examiner Viva Questions & Exact Answers

### Section A: Model Training & Architecture (For Person A)

#### Q1: "What model did you fine-tune, and what is its architecture?"
* **Answer**: We fine-tuned `distilbert-base-uncased`. It has 6 Transformer encoder layers, 12 self-attention heads, and a hidden dimension of 768, totaling ~66 million parameters. It retains $97\%$ of BERT-base's capability while being $40\%$ smaller and $60\%$ faster.

#### Q2: "What is the difference between pre-training and fine-tuning?"
* **Answer**: 
  - **Pre-training** is self-supervised learning on massive unannotated corpora (Wikipedia and BookCorpus) where the model learns general grammar and world knowledge using Masked Language Modeling (MLM).
  - **Fine-tuning** is supervised transfer learning where we freeze or adapt the pre-trained weights and train a task-specific classification head on our labeled interview dataset using a small learning rate ($2 \times 10^{-5}$).

#### Q3: "What is the role of the [CLS] token in DistilBERT?"
* **Answer**: `[CLS]` stands for Classification. Because Transformers use bidirectional self-attention, the `[CLS]` token at position 0 attends to every other token in the sequence. Its final 768-dimensional hidden state acts as an aggregated semantic summary of the entire input pair `[CLS] question [SEP] answer [SEP]`. We feed this vector into our classification head.

#### Q4: "Why did you use AdamW instead of standard Adam?"
* **Answer**: Standard Adam couples L2 weight regularization with gradient moment scaling, which causes weights with large gradients to receive disproportionately less regularization. **AdamW decouples weight decay from the gradient update**, ensuring uniform regularization across all weights and preventing overfitting in Transformer networks.

#### Q5: "Why did you choose CrossEntropyLoss?"
* **Answer**: For multi-class classification across mutually exclusive categories (Poor, Needs Improvement, Good, Excellent), CrossEntropyLoss combines Softmax and Negative Log Likelihood (NLLLoss). It exponentially penalizes confident incorrect predictions, ensuring clean gradient backpropagation.

#### Q6: "What was your dataset size and class distribution?"
* **Answer**: We generated a curated, balanced dataset of 180 interview response pairs across 3 roles (Python Developer, Data Analyst, Web Developer) and 4 quality classes (45 samples per class, exactly $25\%$ each). We used an 80/20 train/validation split (144 train, 36 validation) and achieved $94.44\%$ validation accuracy.

#### Q7: "How does the Attention mechanism work mathematically?"
* **Answer**: The Scaled Dot-Product Attention formula is:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
  $Q$ (Query) represents what a token is looking for, $K$ (Key) represents what a token contains, and $V$ (Value) is the actual content. Dividing by $\sqrt{d_k}$ prevents the dot products from growing excessively large, which would push softmax into regions with vanishing gradients.

---

### Section B: Text Generation & Decoding (For Person A)

#### Q8: "What is autoregressive text generation?"
* **Answer**: An autoregressive model generates text sequentially, one token at a time. Each predicted token is appended to the input sequence and fed back into the model to predict the next token:
  $$P(w_1, w_2, \dots, w_T) = \prod_{t=1}^T P(w_t \mid w_1, \dots, w_{t-1})$$

#### Q9: "What is Greedy decoding, and what is its main flaw?"
* **Answer**: Greedy decoding selects the token with the highest probability at every step: $w_t = \arg\max P(w \mid \text{context})$. Its flaw is that it is myopic—it can miss high-probability sentences that start with a lower-probability first word, and it frequently gets stuck in repetitive loops (e.g., *"and then and then"*).

#### Q10: "How does Temperature scaling work?"
* **Answer**: Temperature $T$ scales the logits before the softmax operation:
  $$P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
  - When $T < 1.0$ (e.g., $0.2$), differences between logits are amplified, making the distribution peaky and deterministic.
  - When $T > 1.0$ (e.g., $1.5$), differences are compressed, making the distribution flatter and more random.

#### Q11: "What is Top-K sampling?"
* **Answer**: Top-K sampling restricts the candidate pool to only the $K$ most probable tokens (e.g., $K=50$). The remaining tokens are masked with $-\infty$. This eliminates the long tail of bizarre tokens while still allowing stochastic diversity.

#### Q12: "What is Top-P (Nucleus) sampling, and why is it better than Top-K?"
* **Answer**: Top-P dynamically chooses the smallest set of tokens whose cumulative probability exceeds threshold $P$ (e.g., $0.90$). It is superior to Top-K because **it adapts to model confidence**:
  - If the model is confident (e.g., *"The capital of France is..."*), the top token might have $95\%$ probability, so the nucleus shrinks to just 1 token.
  - If the model is uncertain, the probability is spread out, so the nucleus automatically expands to 40+ tokens.

#### Q13: "What is a repetition penalty?"
* **Answer**: It penalizes the logits of tokens that have already appeared in the generated sequence. Positive logits are divided by the penalty factor (e.g., $1.2$), and negative logits are multiplied by it, significantly reducing degenerative loops.

---

### Section C: Evaluation Metrics & Prompting (For Person B)

#### Q14: "What is the difference between BLEU and ROUGE?"
* **Answer**: 
  - **BLEU** measures **Precision**: *"Out of the words the candidate generated, how many actually match the reference answer?"*
  - **ROUGE** measures **Recall**: *"Out of the key technical concepts in the reference answer, how many did the candidate successfully capture?"*

#### Q15: "What is the Brevity Penalty in BLEU?"
* **Answer**: If a candidate produces an extremely short answer (e.g., 2 words), precision could be $100\%$ even though the answer is completely inadequate. The Brevity Penalty ($BP = \exp(1 - r/c)$) exponentially penalizes candidate length $c$ when it is shorter than reference length $r$.

#### Q16: "What does ROUGE-L measure?"
* **Answer**: ROUGE-L uses the **Longest Common Subsequence (LCS)** between the candidate and reference texts. Unlike n-grams, LCS does not require words to be strictly consecutive, capturing sentence-level structure and word order similarity.

#### Q17: "What is Chain-of-Thought (CoT) prompting?"
* **Answer**: CoT prompting instructs the model to explicitly output intermediate reasoning steps before arriving at a final conclusion. In SmartHire, we decompose evaluation into:
  1. *Relevance check* (question alignment)
  2. *Concept coverage* (key technical terms)
  3. *Technical depth* (word count and detail)
  4. *Communication quality* (structure and connectors)
  This prevents snap scoring errors and provides actionable feedback.

#### Q18: "What is the Interview Pattern in prompt engineering?"
* **Answer**: The Interview Pattern constrains the LLM to maintain a strict conversational persona. It enforces three rules:
  1. Ask only **one question at a time** (preventing prompt fatigue).
  2. Condition follow-ups on the candidate's prior answer.
  3. Maintain multi-turn transcript history in the prompt context.

#### Q19: "What is Tree-of-Thought (ToT) prompting?"
* **Answer**: Tree-of-Thought extends CoT by exploring multiple reasoning paths simultaneously in a tree structure. In our project, after each answer, the system considers three branches:
  - *Branch A (Probe Weakness)*: Target an area where the candidate scored low.
  - *Branch B (Deepen Strength)*: Challenge the candidate in an area where they showed mastery.
  - *Branch C (New Topic)*: Shift to an unexamined syllabus area.
  The system evaluates candidate momentum and selects the highest-utility branch.

#### Q20: "If the examiner asks: Is this just an API wrapper?"
* **Answer**: 
  > *"No, Sir/Ma'am. All components run completely locally on our machine. We wrote the PyTorch training loop and fine-tuned DistilBERT to classify answer quality with $94.4\%$ accuracy, saving the model weights locally. For text generation, we implemented custom decoding routines—Greedy, Temperature, Top-K, and Top-P—from scratch using DistilGPT-2. Furthermore, our evaluation engine computes mathematical BLEU and ROUGE metrics without any third-party black-box APIs."*

---

## ⚡ Quick 5-Minute Checklist Before You Present

1. Make sure your laptop is plugged into power (so GPU/CPU doesn't throttle).
2. Open terminal in `c:\Users\Rohit\SmartHire\` and run:
   ```bash
   python -m streamlit run app.py
   ```
3. Keep `saved_model/training_curve.png` handy in an image viewer just in case the examiner asks to see the raw plot file.
4. Have your code open in VS Code:
   - Person A should be ready to show `train_evaluator.py` (lines 140–200 for training loop) and `generator.py` (lines 120–230 for Top-K / Top-P sampling).
   - Person B should be ready to show `metrics.py` (lines 60–120 for BLEU calculation) and `prompts.py` (lines 40–120 for Interview Pattern and CoT).
