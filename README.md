# 🎯 SmartHire: AI Mock Interviewer & Answer Evaluation Coach

A complete Generative AI project built with **PyTorch**, **Hugging Face Transformers**, and **Streamlit**. No external paid APIs — everything runs locally on your CPU.

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

## 🛠️ Tech Stack
- **PyTorch** — Model training and inference
- **Hugging Face Transformers** — DistilBERT & DistilGPT-2
- **Streamlit** — Interactive web UI
- **scikit-learn** — Classification metrics
- **NLTK / rouge-score** — NLP evaluation metrics
- **Matplotlib** — Training visualization
