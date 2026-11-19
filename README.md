# 🎯 SmartHire: AI Mock Interviewer & Answer Evaluation Coach

A complete Generative AI project built with **PyTorch**, **Hugging Face Transformers**, and **Streamlit**. No external paid APIs — everything runs locally on your CPU.

---

## 📋 What This Project Demonstrates

| Syllabus Unit | Concept | Where in Project |
|---|---|---|
| Lab 1 & 2 | Fine-tuning on custom dataset | `train_evaluator.py` → DistilBERT classifier |
| Lab 3 | Autoregressive decoding & sampling | `generator.py` → Custom Top-K, Top-P, Temperature |
| Lab 4 | Conditional text generation | `generator.py` → `generate_interview_question()` |
| Unit 2 | Pre-training vs Fine-tuning, Data preprocessing | `train_evaluator.py` → Detailed comments |
| Unit 3 | BLEU & ROUGE evaluation metrics | `metrics.py` → From-scratch BLEU implementation |
| Unit 5 | Interview Pattern, CoT, ToT prompting | `prompts.py` → All 3 techniques |

---

## 🚀 Setup (5 Minutes)

### Step 1: Install Dependencies
```bash
cd SmartHire
pip install -r requirements.txt
```

### Step 2: Generate Training Dataset
```bash
python create_dataset.py
```
This creates `data/interview_dataset.csv` with 180 labeled examples.

### Step 3: Train the Answer Evaluator Model (~5-10 min on CPU)
```bash
python train_evaluator.py
```
This fine-tunes DistilBERT and saves the model to `saved_model/`.

### Step 4: Launch the App
```bash
streamlit run app.py
```

---

## 🏗️ Project Architecture

```
SmartHire/
├── app.py                  # Streamlit web application (4 tabs)
├── generator.py            # Custom autoregressive text generation (Lab 3 & 4)
├── evaluator.py            # Fine-tuned model inference & feedback (Lab 2)
├── train_evaluator.py      # Model fine-tuning pipeline (Lab 1 & 2)
├── metrics.py              # BLEU & ROUGE evaluation metrics (Unit 3)
├── prompts.py              # Prompt engineering templates (Unit 5)
├── create_dataset.py       # Training data generator
├── requirements.txt        # Python dependencies
├── data/
│   ├── question_bank.json  # 45 curated interview questions (3 roles × 3 difficulties × 5 questions)
│   └── interview_dataset.csv  # 180 labeled training examples (auto-generated)
└── saved_model/            # Fine-tuned DistilBERT weights (created after training)
    ├── model.safetensors
    ├── config.json
    ├── tokenizer.json
    ├── training_history.json
    └── training_curve.png
```

---

## 🖥️ App Features (4 Tabs)

### Tab 1: 🎤 Interview Session
- Select job role (Python Developer / Data Analyst / Web Developer)
- AI asks interview questions, you type answers
- Each answer is evaluated by the fine-tuned model in real-time
- Shows score, confidence, BLEU/ROUGE metrics, and CoT reasoning
- Final summary with average score and hiring recommendation

### Tab 2: 🔬 Decoding Lab
- Interactive playground for autoregressive text generation
- Side-by-side comparison of Greedy / Temperature / Top-K / Top-P
- Token-by-token probability visualization
- Adjustable sliders for all sampling parameters

### Tab 3: 📊 Evaluation Metrics
- Paste any reference + candidate text
- Computes BLEU-1 through BLEU-4 and ROUGE-1/2/L
- Visual interpretation of precision vs recall

### Tab 4: 🧠 Training Results
- Training loss curve visualization
- Epoch-by-epoch accuracy and loss
- Model architecture diagram
- Quick test interface for the classifier

---

## 👥 Division of Work (2 People)

### Student A: The Model & Generation Lead
- **Explains**: Autoregressive decoding, next-token prediction, Attention mechanism
- **Demos**: Tab 2 (Decoding Lab) — shows Greedy vs Top-K vs Top-P differences
- **Covers**: `generator.py`, `train_evaluator.py`, `evaluator.py`

### Student B: The Evaluation & Prompts Lead
- **Explains**: BLEU vs ROUGE (Precision vs Recall), Chain-of-Thought reasoning
- **Demos**: Tab 1 (Interview) & Tab 3 (Metrics) — shows evaluation pipeline
- **Covers**: `metrics.py`, `prompts.py`, `app.py`

---

## 🎓 Top 15 Viva Questions & Answers

1. **What is autoregressive decoding?**
   > The model predicts one token at a time, feeds it back as input, and predicts the next: P(w₁,...,wₜ) = ∏ P(wₜ | w₁,...,wₜ₋₁)

2. **What is Temperature in sampling?**
   > It scales logits before softmax. Lower temperature (0.2) makes output deterministic; higher (1.5) makes it creative but potentially incoherent.

3. **What is the difference between Top-K and Top-P?**
   > Top-K keeps a fixed number of top tokens. Top-P (Nucleus) dynamically adjusts based on cumulative probability — better because it adapts to model confidence.

4. **What is fine-tuning vs pre-training?**
   > Pre-training learns general language from massive data (Wikipedia). Fine-tuning adapts those weights to a specific task with a small labeled dataset.

5. **Why use AdamW instead of Adam?**
   > AdamW decouples weight decay from the gradient update, ensuring uniform regularization across all parameters. Standard Adam applies uneven decay.

6. **What is the [CLS] token?**
   > A special token at position 0 whose final hidden state summarizes the entire input. We attach a classification head on top of it.

7. **What does BLEU measure?**
   > Precision — "Of the words the model generated, how many match the reference?"

8. **What does ROUGE measure?**
   > Recall — "Of the words in the reference, how many did the model capture?"

9. **What is Chain-of-Thought prompting?**
   > Forcing the model to reason step-by-step before giving a final answer. This reduces errors on complex tasks by decomposing reasoning.

10. **What is the Interview Pattern?**
    > A prompt engineering technique that constrains the LLM into a specific persona (interviewer), maintaining multi-turn conversational structure.

11. **What is Tree-of-Thought?**
    > An extension of CoT that explores multiple reasoning paths simultaneously, then selects the best one — like a search tree over hypotheses.

12. **Why CrossEntropyLoss for classification?**
    > It combines Softmax + NLLLoss in a numerically stable way, and penalizes confident wrong predictions exponentially.

13. **Why DistilBERT and not full BERT?**
    > DistilBERT has 40% fewer parameters, is 60% faster, and retains 97% of BERT's accuracy. Perfect for CPU-based student projects.

14. **What is the Attention mechanism?**
    > Every token computes Query, Key, Value vectors. Attention(Q,K,V) = softmax(QK^T / √d) · V. This lets each word weigh the importance of every other word.

15. **Is this just an API wrapper?**
    > No. We fine-tuned DistilBERT locally using PyTorch, implemented custom autoregressive decoding from scratch, and computed BLEU/ROUGE manually. All model weights are local.

---

## 🛠️ Tech Stack
- **PyTorch** — Model training and inference
- **Hugging Face Transformers** — DistilBERT & DistilGPT-2
- **Streamlit** — Interactive web UI
- **scikit-learn** — Classification metrics
- **NLTK / rouge-score** — NLP evaluation metrics
- **Matplotlib** — Training visualization
