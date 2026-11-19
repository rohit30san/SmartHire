"""
===================================================================================
SmartHire - Automated Interview Answer Evaluator: Fine-Tuning Pipeline
===================================================================================
File: train_evaluator.py
Purpose: Fine-tunes a pre-trained DistilBERT model for multi-class sequence
         classification to evaluate candidate interview answers into 4 quality tiers:
         0 = Poor, 1 = Needs Improvement, 2 = Good, 3 = Excellent.

-----------------------------------------------------------------------------------
VIVA PREPARATION GUIDE - KEY CONCEPTS & THEORETICAL FOUNDATIONS
-----------------------------------------------------------------------------------
1. PRE-TRAINING VS. FINE-TUNING:
   - Pre-training: Self-supervised learning on massive unannotated corpora (e.g.,
     Wikipedia and BookCorpus, ~3.3 billion words). The model solves general
     language tasks like Masked Language Modeling (predicting hidden tokens) and
     learns foundational syntax, semantics, context, and world representations.
     This phase requires massive compute (weeks on clusters of TPUs/GPUs).
   - Fine-tuning: Supervised transfer learning where the pre-trained weights serve
     as an initialization for a specific downstream task (here, evaluating interview
     responses). A task-specific classification head (linear projection) is placed
     on top of the pooled [CLS] representation. We adjust all or top layers using a
     small learning rate (e.g., 2e-5) on our curated dataset (~180 labeled samples).
     This takes only 5-10 minutes on CPU and achieves high task accuracy without
     requiring millions of labeled training instances.

2. WHY AdamW INSTEAD OF REGULAR Adam:
   - In standard Adam (Adaptive Moment Estimation), L2 weight regularization is
     added directly to the objective loss function: L_total = L + 0.5 * lambda * ||w||^2.
   - When gradients are computed, Adam divides the update step by the running estimate
     of past squared gradients (sqrt(v_t) + eps). Consequently, weights that undergo
     frequent, large updates receive disproportionately LESS weight decay penalty,
     whereas weights with infrequent small updates receive MORE penalty!
   - AdamW (Loshchilov & Hutter, 2017) decouples weight decay from the adaptive gradient
     update: w_t = w_(t-1) - alpha * (m_t / (sqrt(v_t) + eps)) - alpha * lambda * w_(t-1).
   - This ensures uniform weight shrinkage across all parameters, drastically improving
     generalization and preventing overfitting in Transformer architectures.

3. WHY CrossEntropyLoss FOR MULTI-CLASS CLASSIFICATION:
   - For multi-class classification across mutually exclusive categories (Poor,
     Needs Improvement, Good, Excellent), CrossEntropyLoss calculates the negative
     log-likelihood of the correct class:
     L = -log( p_target ) = -log( exp(z_target) / sum(exp(z_j)) )
   - Key mathematical advantages:
     a) It combines Softmax and Negative Log Likelihood (NLLLoss) in a single
        numerically stable operation using the Log-Sum-Exp trick.
     b) Penalizes confident wrong predictions exponentially (approaches infinity as
        predicted probability for true class approaches 0).
     c) Its gradient with respect to logit z_i simplifies cleanly to (p_i - y_i),
        providing non-saturating, smooth gradient flow throughout backpropagation.

4. WHAT IS THE [CLS] TOKEN AND WHY DO WE USE IT:
   - In BERT and DistilBERT architectures, '[CLS]' (Classification) is a unique
     sentinel token prepended to position 0 of every input sequence.
   - Transformers employ full bidirectional multi-head self-attention, meaning every
     token can attend to every other token across all hidden layers.
   - Because the [CLS] token does not correspond to any specific word in the text, its
     final hidden state representation (h_CLS in R^768) acts as an aggregated, pooled
     summary embedding of the entire sentence/pair.
   - Feeding h_CLS through a classification head (Dropout -> Linear -> 4 classes)
     yields an invariant, fixed-size vector regardless of input sequence length.
   - Sequence format: '[CLS] {question} [SEP] {answer} [SEP]'
     Here, '[SEP]' explicitly delimits the question boundary from the candidate's
     response, allowing attention heads to compute cross-attention between what the
     question demands and what the answer delivers.
===================================================================================
"""

import os
import sys
import json
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ---------------------------------------------------------------------------------
# 1. REPRODUCIBILITY & DEVICE SETUP
# ---------------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic, reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Set seed globally
set_seed(42)

# Select computing device (CPU-safe)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label mappings
LABEL_TO_ID = {
    "Poor": 0,
    "Needs Improvement": 1,
    "Good": 2,
    "Excellent": 3
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


# ---------------------------------------------------------------------------------
# 2. DATASET LOADER & FALLBACK GENERATOR
# ---------------------------------------------------------------------------------
def generate_synthetic_dataset(output_path: str) -> pd.DataFrame:
    """
    Generates a realistic, comprehensive interview dataset (~180 examples) across
    4 quality tiers (Poor, Needs Improvement, Good, Excellent) covering Python,
    System Design, Data Structures, Machine Learning, and Behavioral questions.
    Ensures train_evaluator.py is 100% self-contained and runnable out-of-the-box.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    questions_and_answers = [
        # 1. Python GIL
        {
            "q": "What is the Python Global Interpreter Lock (GIL) and how does it impact multi-threading?",
            "ans": [
                ("The GIL is a lock in Python that stops bugs.", 0),
                ("The GIL allows only one thread to run at a time in Python, so multi-threading doesn't make code faster.", 1),
                ("The Global Interpreter Lock (GIL) is a mutex in CPython that ensures only one thread executes Python bytecode at a time. It prevents race conditions in memory management but limits CPU-bound multi-threading.", 2),
                ("The Global Interpreter Lock (GIL) is a synchronization mutex in CPython designed to manage thread-safe reference counting. It restricts bytecode execution to a single thread at any instant, making standard multithreading ineffective for CPU-bound tasks. To bypass this limitation, developers use the multiprocessing module, C-extensions, or alternative implementations like PyPy-STM or Jython.", 3)
            ]
        },
        # 2. Python Generators
        {
            "q": "Explain Python generators and how the yield statement works.",
            "ans": [
                ("Generators generate numbers in a loop using yield.", 0),
                ("Generators return items one by one instead of all at once to save memory using the yield keyword.", 1),
                ("A generator is a function that produces a sequence of values lazily using the yield statement. When yield is encountered, execution state is frozen and resumed on the next call to next().", 2),
                ("Generators are memory-efficient iterators implemented using generator functions or expressions. Unlike standard functions that return a complete list and terminate, a generator yields values lazily on-demand. When 'yield' is executed, the interpreter preserves local variable states and instruction pointers, consuming O(1) auxiliary memory. This makes them ideal for processing large datasets or infinite streams.", 3)
            ]
        },
        # 3. Mutable vs Immutable
        {
            "q": "What is the difference between mutable and immutable data types in Python?",
            "ans": [
                ("Mutable can change, immutable cannot change.", 0),
                ("Lists and dictionaries are mutable because you can modify them. Strings and tuples are immutable because their values cannot be edited after creation.", 1),
                ("In Python, mutable objects (like lists, dicts, sets) can alter their internal state in-place without changing their memory address (id). Immutable objects (integers, strings, tuples) cannot be altered; modifying them allocates a new object in memory.", 2),
                ("Mutability in Python defines whether an object's memory state can be modified post-instantiation. Mutable types (list, dict, set) allow in-place mutations, preserving object identity (id()). Immutable types (str, int, tuple, frozenset) raise errors upon mutation attempts; re-assignment creates a new object in memory. This distinction is critical for hashability: only immutable objects are hashable and valid as dictionary keys or set elements.", 3)
            ]
        },
        # 4. REST vs GraphQL
        {
            "q": "Compare REST APIs with GraphQL. What are the key architectural differences?",
            "ans": [
                ("REST uses URLs and GraphQL uses queries.", 0),
                ("REST has multiple endpoints for different data while GraphQL uses a single endpoint where clients request specific fields.", 1),
                ("REST is an architectural style based on HTTP methods (GET, POST, PUT, DELETE) and fixed resources per endpoint, which can cause over-fetching or under-fetching. GraphQL is a query language providing a single endpoint where clients declare the exact fields required.", 2),
                ("REST is resource-centric and relies on standard HTTP methods across decentralized URI endpoints, often leading to over-fetching or multiple round-trip requests (under-fetching). In contrast, GraphQL utilizes a single POST endpoint with a strongly typed schema (SDL), enabling clients to request exact sub-fields in a single network round-trip. While GraphQL minimizes payload size and round-trips, REST provides superior out-of-the-box HTTP caching, CDN integration, and simpler rate limiting.", 3)
            ]
        },
        # 5. SQL vs NoSQL
        {
            "q": "When would you choose a NoSQL database over a relational SQL database?",
            "ans": [
                ("NoSQL is faster and newer than SQL databases.", 0),
                ("Choose NoSQL when you have unstructured data, need horizontal scaling, or have flexible JSON-like documents.", 1),
                ("SQL databases (PostgreSQL, MySQL) excel at structured data with strict ACID compliance and complex relational joins. Choose NoSQL (MongoDB, Cassandra) when dealing with schema flexibility, unstructured payloads, high write throughput, or horizontal partitioning across clusters.", 2),
                ("The choice depends on the CAP theorem and domain constraints. Relational SQL databases guarantee ACID transactions and relational integrity, ideal for financial and enterprise domains requiring complex multi-table joins. NoSQL (document, key-value, column-family, graph) is preferable when horizontal sharding, partition tolerance (AP/CP), polymorphic schemas, and rapid horizontal scaling across distributed nodes take precedence over strict immediate consistency.", 3)
            ]
        },
        # 6. Overfitting in ML
        {
            "q": "What is overfitting in machine learning and how do you prevent it?",
            "ans": [
                ("Overfitting is when the model is bad and gives wrong answers.", 0),
                ("Overfitting occurs when a model learns the training data too well and performs poorly on new test data. You can fix it with more data or dropout.", 1),
                ("Overfitting is high variance where a model memorizes noise and specific patterns of the training set rather than generalizing. Prevention techniques include L1/L2 regularization, dropout, data augmentation, early stopping, and cross-validation.", 2),
                ("Overfitting occurs when a high-capacity model minimizes empirical training loss by learning stochastic noise rather than underlying data distributions, manifesting as low training error but high generalization error. Mitigation strategies include regularization (L1 Lasso for sparsity, L2 Ridge/weight decay), stochastic dropout, cross-validation, data augmentation, pruning, ensemble learning (Random Forests), and early stopping based on validation loss.", 3)
            ]
        },
        # 7. Hash Table Collision Resolution
        {
            "q": "How do hash tables handle collisions? Compare separate chaining with open addressing.",
            "ans": [
                ("Hash tables use hashes so collisions don't happen often.", 0),
                ("Collisions happen when two keys hash to the same bucket. You can use linked lists (chaining) or find the next open slot.", 1),
                ("A collision occurs when hash(key1) == hash(key2). In separate chaining, each bucket holds a linked list or red-black tree of colliding entries. In open addressing (linear probing, quadratic probing, double hashing), colliding items are placed in alternate available slots within the table array itself.", 2),
                ("Hash collisions are resolved primarily via separate chaining or open addressing. In separate chaining, each array bucket stores a pointer to a secondary data structure (linked list or balanced BST, as in Java 8 HashMap), gracefully handling load factors > 1.0. In open addressing, all elements reside within the array itself; collisions trigger probing sequences (linear, quadratic, or double hashing). Open addressing provides better cache locality but suffers from clustering and degrades rapidly as the load factor exceeds 0.7.", 3)
            ]
        },
        # 8. Microservices vs Monolith
        {
            "q": "What are the trade-offs between monolithic and microservice architectures?",
            "ans": [
                ("Monoliths are old and bad, microservices are modern and fast.", 0),
                ("Monoliths are single codebases that are easy to deploy initially. Microservices break services into independent parts for scalability but are more complex.", 1),
                ("Monoliths offer simple development, debugging, and atomic transactions but scale as a single unit. Microservices enable independent deployments, technology diversity, and localized scaling, but introduce distributed system complexities like network latency, distributed tracing, and data consistency issues.", 2),
                ("A monolith is a unified deployment unit that simplifies end-to-end testing, local debugging, and ACID compliance, but suffers from tight coupling, long build times, and blast-radius vulnerability. Microservices decompose functionality into independently deployable, bounded contexts (DDD), enabling autonomous team velocity and targeted horizontal scaling. However, they introduce significant distributed systems overhead: partial failures, eventual consistency (Saga pattern), complex observability, and inter-service network latency.", 3)
            ]
        },
        # 9. QuickSort vs MergeSort
        {
            "q": "Explain the differences between QuickSort and MergeSort. When would you use one over the other?",
            "ans": [
                ("Both are sorting algorithms that run in O(n log n) time.", 0),
                ("QuickSort picks a pivot and partitions the array. MergeSort splits the array in half and merges sorted halves. MergeSort takes extra memory.", 1),
                ("QuickSort is an in-place, unstable divide-and-conquer algorithm with average O(n log n) and worst-case O(n^2) time complexity. MergeSort is a stable divide-and-conquer algorithm that guarantees O(n log n) time but requires O(n) auxiliary space. QuickSort is preferred for arrays due to cache locality, while MergeSort is preferred for linked lists and external sorting.", 2),
                ("QuickSort partitions arrays in-place around a pivot element with average O(n log n) time and O(log n) auxiliary stack space, but degrades to O(n^2) under adversarial pivot selections. MergeSort recursively bisects inputs and merges sorted sub-arrays, guaranteeing deterministic O(n log n) runtime and stability, but incurring O(n) auxiliary memory overhead. In practice, QuickSort (e.g., dual-pivot) is preferred for contiguous in-memory arrays due to superior CPU cache locality, while MergeSort is favored for linked structures, objects requiring stable sorting, and external tape/disk sorting.", 3)
            ]
        },
        # 10. Behavioral: Handling Deadline Pressure
        {
            "q": "Tell me about a time you faced a tight project deadline. How did you prioritize tasks?",
            "ans": [
                ("I just worked overtime and drank coffee to finish it.", 0),
                ("I had a project due soon, so I made a to-do list, focused on the main features, and delivered it on time.", 1),
                ("During a critical product launch, our team had two weeks to deliver a payment gateway integration. I held an emergency triage session, identified core MVP user journeys, deferred non-critical UI animations to phase two, and instituted daily 15-minute syncs to remove blockers, successfully shipping on schedule.", 2),
                ("In my previous role, a client required an unexpected compliance feature within 10 business days. Applying the STAR framework: Situation: Non-negotiable regulatory deadline risking contract renewal. Task: Implement encrypted audit logging across 4 microservices. Action: I mapped requirements using the MoSCoW prioritization framework, isolated must-have logging endpoints from nice-to-have dashboard visualizations, automated integration tests to prevent regression, and coordinated pair programming on critical paths. Result: We passed compliance review with zero defects 24 hours ahead of deadline.", 3)
            ]
        },
        # 11. Database Indexing
        {
            "q": "How does database indexing work, and what is the trade-off of adding indexes?",
            "ans": [
                ("Indexes make queries fast by looking up things.", 0),
                ("An index is like a book index. It helps find rows faster without scanning the whole table, but makes inserts a bit slower.", 1),
                ("Database indexes (commonly B-Trees) store sorted references to table rows, reducing query lookup complexity from O(N) full table scans to O(log N). The trade-off is increased storage overhead and slower write operations (INSERT, UPDATE, DELETE) since indexes must be updated synchronously.", 2),
                ("Database indexes utilize self-balancing B+ Trees or hash tables to map key column values to physical disk tuple locators. By maintaining sorted leaf nodes with bidirectional pointers, B+ Trees transform O(N) sequential table scans into O(log N) tree traversals and enable efficient range queries. The trade-off is dual-pronged: physical disk storage overhead and write amplification. Every INSERT/UPDATE/DELETE incurs synchronous index tree rebalancing and page splits, meaning excessive indexing degrades write throughput.", 3)
            ]
        },
        # 12. Cross-Validation
        {
            "q": "What is K-Fold cross-validation and why is it preferred over a simple train-test split?",
            "ans": [
                ("It splits data K times so you get better accuracy.", 0),
                ("K-Fold splits data into K equal parts. You train on K-1 folds and test on the remaining fold, repeating K times to get an average score.", 1),
                ("K-Fold cross-validation partitions the dataset into K subsets. The model trains on K-1 folds and validates on the held-out fold across K iterations. It provides a more robust estimate of generalization performance and prevents data selection bias inherent in a single train-test split.", 2),
                ("In K-Fold cross-validation, the dataset is partitioned into K non-overlapping, equally sized subsets. In K consecutive iterations, K-1 folds serve as the training partition while the remaining fold evaluates validation error. The final metric is the arithmetic mean across all K runs. Unlike a single holdout split—which is susceptible to high variance and sampling bias, especially on small-to-moderate sample sizes—K-Fold ensures every observation is evaluated exactly once in a test partition, yielding lower estimator variance and reliable hyperparameter tuning.", 3)
            ]
        }
    ]
    
    rows = []
    # Repeat with slight linguistic variations to reach ~180 diverse examples
    variations = [
        "",
        " Specifically,",
        " In practical software engineering,",
        " From an engineering standpoint,",
    ]
    
    for v_idx, var in enumerate(variations):
        for item in questions_and_answers:
            q = item["q"]
            for ans_text, label_id in item["ans"]:
                if v_idx > 0 and label_id >= 2:
                    ans_text = f"{ans_text}{var} this concept is critical in production environments."
                rows.append({
                    "question": q,
                    "answer": ans_text,
                    "label": label_id,
                    "label_name": ID_TO_LABEL[label_id]
                })
    
    # Shuffle dataset
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"Generated synthetic training dataset with {len(df)} samples at: {output_path}")
    return df


def load_interview_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Loads interview_dataset.csv from data directory. If missing, automatically
    creates the dataset so that training executes seamlessly without manual steps.
    """
    candidates = [
        os.path.join(data_dir, "interview_dataset.csv"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), data_dir, "interview_dataset.csv"),
        os.path.join(os.getcwd(), data_dir, "interview_dataset.csv"),
    ]
    
    target_csv = None
    for path in candidates:
        if os.path.exists(path):
            target_csv = path
            break
            
    if target_csv is None:
        target_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "interview_dataset.csv")
        qb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "question_bank.json")
        if os.path.exists(qb_path):
            try:
                print(f"Generating full dataset from {qb_path} using create_dataset...")
                import create_dataset
                create_dataset.generate_full_dataset(qb_path, target_csv, seed=42)
                df = pd.read_csv(target_csv)
            except Exception as e:
                print(f"Fallback to internal generator due to: {e}")
                df = generate_synthetic_dataset(target_csv)
        else:
            print(f"Dataset not found at expected locations. Creating dataset at {target_csv}...")
            df = generate_synthetic_dataset(target_csv)
    else:
        print(f"Loading existing dataset from: {target_csv}")
        df = pd.read_csv(target_csv)
        
    # Standardize column names (case-insensitive)
    col_map = {col.lower().strip(): col for col in df.columns}
    q_col = col_map.get("question", "question")
    a_col = col_map.get("answer", "answer")
    l_col = None
    for candidate in ["label", "label_id", "quality", "score", "rating"]:
        if candidate in col_map:
            l_col = col_map[candidate]
            break
            
    if l_col is None:
        raise ValueError("Could not find label column in CSV. Expected 'label', 'label_id', or 'quality'.")
        
    # Map labels to integer IDs 0..3 if formatted as text
    def map_label(val):
        if isinstance(val, (int, np.integer)):
            return int(val)
        val_str = str(val).strip().title()
        if val_str in LABEL_TO_ID:
            return LABEL_TO_ID[val_str]
        try:
            return int(val)
        except ValueError:
            return 0
            
    cleaned_df = pd.DataFrame({
        "question": df[q_col].astype(str),
        "answer": df[a_col].astype(str),
        "label": df[l_col].apply(map_label).astype(int)
    })
    
    print(f"Loaded {len(cleaned_df)} examples across 4 classes:")
    for lbl_id, lbl_name in ID_TO_LABEL.items():
        count = (cleaned_df["label"] == lbl_id).sum()
        print(f"  Class {lbl_id} ({lbl_name}): {count} samples")
        
    return cleaned_df


# ---------------------------------------------------------------------------------
# 3. PYTORCH DATASET CLASS
# ---------------------------------------------------------------------------------
class InterviewDataset(Dataset):
    """
    PyTorch Dataset for Interview Question & Answer sequence classification.
    
    Preprocesses inputs by concatenating Question and Answer with special tokens:
    '[CLS] {question} [SEP] {answer} [SEP]'
    
    Returns tokenized tensors: input_ids, attention_mask, and labels.
    """
    def __init__(self, questions, answers, labels, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = list(labels)
        
        # Combine question and answer into sequence pair representation
        self.formatted_texts = [
            f"[CLS] {str(q).strip()} [SEP] {str(a).strip()} [SEP]"
            for q, a in zip(questions, answers)
        ]
        
    def __len__(self) -> int:
        return len(self.formatted_texts)
        
    def __getitem__(self, idx: int) -> dict:
        text = self.formatted_texts[idx]
        label = self.labels[idx]
        
        # add_special_tokens=False because [CLS] and [SEP] are explicitly in the text string
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),        # [max_length]
            "attention_mask": encoding["attention_mask"].squeeze(0), # [max_length]
            "labels": torch.tensor(label, dtype=torch.long)
        }


# ---------------------------------------------------------------------------------
# 4. TRAINING & EVALUATION FUNCTIONS
# ---------------------------------------------------------------------------------
def train_epoch(model, dataloader, optimizer, criterion, device):
    """Performs one complete training epoch over the training dataloader."""
    model.train()
    total_loss = 0.0
    correct_preds = 0
    total_samples = 0
    
    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Zero parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Compute loss
        loss = criterion(logits, labels)
        
        # Backward pass & parameter update
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item() * len(labels)
        preds = torch.argmax(logits, dim=1)
        correct_preds += (preds == labels).sum().item()
        total_samples += len(labels)
        
    avg_loss = total_loss / total_samples
    accuracy = correct_preds / total_samples
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on validation data without gradient computations."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, all_preds, all_labels


# ---------------------------------------------------------------------------------
# 5. MAIN TRAINING PIPELINE
# ---------------------------------------------------------------------------------
def train_model(
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    max_length: int = 256,
    save_dir: str = "saved_model"
):
    """
    Main orchestration function:
    1. Loads dataset
    2. Initializes DistilBERT tokenizer & model
    3. Splits 80/20 train/validation
    4. Executes training loop with AdamW and CrossEntropyLoss
    5. Saves weights, history, loss curves, and evaluation reports
    """
    start_time = time.time()
    print("=" * 70)
    print("SMARTHIRE: DISTILBERT ANSWER EVALUATOR FINE-TUNING")
    print(f"Device: {DEVICE} (CUDA Available: {torch.cuda.is_available()})")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {learning_rate}")
    print("=" * 70)
    
    # Load dataset
    df = load_interview_data("data")
    
    # Stratified 80/20 train/validation split
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )
    print(f"Dataset split: {len(train_df)} training samples | {len(val_df)} validation samples")
    
    # Initialize DistilBERT Tokenizer
    model_name = "distilbert-base-uncased"
    print(f"\nLoading pre-trained tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create PyTorch Datasets and DataLoaders
    train_dataset = InterviewDataset(
        train_df["question"].values,
        train_df["answer"].values,
        train_df["label"].values,
        tokenizer,
        max_length=max_length
    )
    val_dataset = InterviewDataset(
        val_df["question"].values,
        val_df["answer"].values,
        val_df["label"].values,
        tokenizer,
        max_length=max_length
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Load DistilBERT Sequence Classification Model (4 labels)
    print(f"Loading pre-trained model: {model_name} (num_labels=4)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=4,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID
    )
    model.to(DEVICE)
    
    # Setup Optimizer & Loss function
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    
    # Track metrics history
    history = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }
    
    print("\nStarting Training Loop...")
    print("-" * 70)
    print(f"{'Epoch':<8}{'Train Loss':<15}{'Train Acc (%)':<16}{'Val Loss':<15}{'Val Acc (%)':<15}")
    print("-" * 70)
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Train epoch
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        
        # Validate epoch
        val_loss, val_acc, val_preds, val_targets = evaluate(model, val_loader, criterion, DEVICE)
        
        epoch_duration = time.time() - epoch_start
        
        # Log metrics
        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["train_accuracy"].append(round(train_acc * 100, 2))
        history["val_loss"].append(round(val_loss, 4))
        history["val_accuracy"].append(round(val_acc * 100, 2))
        
        print(f"{epoch}/{epochs:<6}{train_loss:<15.4f}{train_acc * 100:<16.2f}{val_loss:<15.4f}{val_acc * 100:<15.2f} ({epoch_duration:.1f}s)")
        
    print("-" * 70)
    total_time = time.time() - start_time
    print(f"Training completed in {total_time / 60:.2f} minutes.")
    
    # Final validation pass to produce classification report
    final_val_loss, final_val_acc, final_preds, final_targets = evaluate(
        model, val_loader, criterion, DEVICE
    )
    
    print("\n" + "=" * 70)
    print("FINAL EVALUATION METRICS ON VALIDATION SET")
    print("=" * 70)
    print(f"Final Validation Accuracy: {final_val_acc * 100:.2f}%\n")
    target_names = [ID_TO_LABEL[i] for i in range(4)]
    report = classification_report(
        final_targets,
        final_preds,
        target_names=target_names,
        digits=4,
        zero_division=0
    )
    print("Classification Report:")
    print(report)
    
    # -----------------------------------------------------------------------------
    # 6. ARTIFACT SAVING & PLOTTING
    # -----------------------------------------------------------------------------
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_dir)
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\nSaving model weights and tokenizer to: {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Save training history JSON
    history_file = os.path.join(save_path, "training_history.json")
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
    print(f"Saved training history metrics: {history_file}")
    
    # Generate and save training curves plot
    plot_path = os.path.join(save_path, "training_curve.png")
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], "o-", label="Training Loss", color="#1f77b4", linewidth=2)
    plt.plot(history["epoch"], history["val_loss"], "s--", label="Validation Loss", color="#ff7f0e", linewidth=2)
    plt.title("Cross-Entropy Loss vs. Epochs", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], history["train_accuracy"], "o-", label="Train Accuracy", color="#2ca02c", linewidth=2)
    plt.plot(history["epoch"], history["val_accuracy"], "^--", label="Val Accuracy", color="#d62728", linewidth=2)
    plt.title("Classification Accuracy vs. Epochs", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved training curve visualization: {plot_path}")
    print("=" * 70)
    print("Model fine-tuning and export completed successfully!")
    print("=" * 70)


# ---------------------------------------------------------------------------------
# 7. SCRIPT ENTRY POINT
# ---------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on Interview Evaluation Dataset.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay (default: 0.01)")
    parser.add_argument("--save_dir", type=str, default="saved_model", help="Directory to save model weights")
    args = parser.parse_args()
    
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir
    )
