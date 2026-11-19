"""
===================================================================================
SmartHire - Automated Interview Answer Evaluator: Inference & Feedback Module
===================================================================================
File: evaluator.py
Purpose: Production inference module for evaluating interview response quality
         using fine-tuned DistilBERT sequence classification. Provides Chain-of-Thought
         rule-based feedback and robust fallback handling if model weights are not yet trained.

-----------------------------------------------------------------------------------
VIVA PREPARATION GUIDE - MODEL ARCHITECTURE & INFERENCE
-----------------------------------------------------------------------------------
1. INFERENCE WORKFLOW:
   Input (Question + Candidate Answer)
      |
      v
   Text Formatting: '[CLS] {question} [SEP] {answer} [SEP]'
      |
      v
   DistilBERT Tokenizer (max_length=256, padding/truncation, WordPiece vocabulary)
      |
      v
   DistilBERT Transformer Encoder (6 layers, 12 attention heads, 768 hidden dimension)
      |
      v
   Pooled [CLS] Representation (h_CLS in R^768 at token index 0)
      |
      v
   Pre-classifier Dropout (p=0.2) -> Linear Classifier Layer (768 -> 4 logits)
      |
      v
   Softmax Activation -> Class Probabilities (Poor, Needs Improvement, Good, Excellent)
      |
      v
   Expected Percentage Score (0-100) + Rule-Based Chain-of-Thought Feedback Generation

2. KEY ADVANTAGES OF DistilBERT OVER STANDARD BERT:
   - DistilBERT is distilled via Knowledge Distillation from BERT-base (Sanh et al., 2019).
   - Retains 97% of BERT's language understanding performance on GLUE benchmark.
   - 40% fewer parameters (66M vs 110M) and 60% faster inference speed.
   - Operates efficiently on standard CPUs without requiring expensive dedicated GPUs.
===================================================================================
"""

import os
import re
import string
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABEL_MAP = {0: 'Poor', 1: 'Needs Improvement', 2: 'Good', 3: 'Excellent'}
ID_TO_LABEL = LABEL_MAP
LABEL_TO_ID = {v: k for k, v in LABEL_MAP.items()}

# Common English stop words for keyword extraction
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}


class AnswerEvaluator:
    """Loads the fine-tuned DistilBERT model for answer quality evaluation.
    
    VIVA EXPLANATION:
    This class performs INFERENCE using our fine-tuned model.
    Fine-tuning = taking a pre-trained language model (DistilBERT, which was
    pre-trained on Wikipedia/BookCorpus using Masked Language Modeling) and
    training it further on our specific task (interview answer quality classification).
    
    The model architecture:
    Input text -> DistilBERT Encoder -> [CLS] token embedding -> Linear classifier -> 4 classes
    """

    def __init__(self, model_path: str = 'saved_model'):
        """Load the fine-tuned model and tokenizer.
        
        If the model directory does not exist or lacks model weights, falls back
        gracefully to rule-based keyword evaluation mode with clear user guidance.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fallback_mode = False
        self.model = None
        self.tokenizer = None
        
        # Resolve potential model path candidates
        candidate_paths = [
            model_path,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path),
            os.path.join(os.getcwd(), model_path)
        ]
        
        resolved_path = None
        for path in candidate_paths:
            # Check for standard model files (config.json + model weights)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, 'config.json')):
                resolved_path = path
                break
                
        if resolved_path is None:
            print("\n" + "=" * 76)
            print("[NOTICE] Fine-tuned model directory not found at: '{}'".format(model_path))
            print("To train the full neural model, please run:")
            print("    python train_evaluator.py")
            print("SmartHire is activating intelligent KEYWORD-BASED EVALUATION MODE.")
            print("You can evaluate responses, test UI features, and demo the application!")
            print("=" * 76 + "\n")
            self.fallback_mode = True
        else:
            try:
                print(f"Loading fine-tuned DistilBERT evaluator from: {resolved_path}...")
                self.tokenizer = AutoTokenizer.from_pretrained(resolved_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(resolved_path)
                self.model.to(self.device)
                self.model.eval()
                self.fallback_mode = False
                print(f"Model successfully loaded on {self.device}. Ready for real-time inference.")
            except Exception as e:
                print(f"[WARNING] Failed to load model weights: {e}")
                print("Falling back to keyword-based evaluation mode.")
                self.fallback_mode = True

    def _extract_keywords(self, text: str) -> set:
        """Helper to extract non-stopword alphanumeric tokens from text."""
        clean_text = text.lower()
        clean_text = re.sub(r'[^\w\s]', ' ', clean_text)
        tokens = clean_text.split()
        return {w for w in tokens if len(w) > 2 and w not in STOP_WORDS}

    def _fallback_evaluate(self, question: str, answer: str) -> dict:
        """
        Intelligent rule-based fallback evaluation when trained model weights
        are not yet generated on the machine. Analyzes length, technical vocabulary,
        keyword overlap, and structural depth.
        """
        q_clean = question.strip()
        a_clean = answer.strip()
        
        # Word count metrics
        words = a_clean.split()
        word_count = len(words)
        
        # Keyword overlap
        q_keywords = self._extract_keywords(q_clean)
        a_keywords = self._extract_keywords(a_clean)
        overlap = q_keywords.intersection(a_keywords)
        overlap_ratio = len(overlap) / max(1, len(q_keywords))
        
        # Technical depth indicators
        tech_indicators = {
            "complexity", "o(1)", "o(n)", "o(n log n)", "o(log n)", "trade-off",
            "tradeoff", "latency", "memory", "overhead", "for example", "e.g.",
            "such as", "specifically", "architecture", "in-place", "concurrency",
            "thread", "asynchronous", "scalability", "implementation", "because",
            "therefore", "advantage", "disadvantage", "caching", "index", "b-tree",
            "star", "situation", "task", "action", "result"
        }
        found_tech = [term for term in tech_indicators if term in a_clean.lower()]
        tech_score = min(len(found_tech) * 12, 40)
        
        # Length scoring (target: 40-120 words for good/excellent answers)
        if word_count < 10:
            length_score = 10
        elif word_count < 25:
            length_score = 30
        elif word_count < 55:
            length_score = 65
        elif word_count < 130:
            length_score = 85
        else:
            length_score = 90
            
        # Composite heuristic score (0-100)
        raw_score = int(0.35 * length_score + 0.35 * (overlap_ratio * 100) + 0.30 * tech_score)
        score = max(8, min(97, raw_score))
        
        # Determine classification tier
        if score < 40 or word_count < 8:
            label_id = 0
        elif score < 65:
            label_id = 1
        elif score < 85:
            label_id = 2
        else:
            label_id = 3
            
        label = LABEL_MAP[label_id]
        
        # Generate normalized pseudo-probabilities centered at the predicted label
        base_probs = [0.08, 0.12, 0.18, 0.12]
        base_probs[label_id] += 0.50
        total = sum(base_probs)
        normalized_probs = [round(p / total, 4) for p in base_probs]
        confidence = normalized_probs[label_id]
        
        prob_dict = {LABEL_MAP[i]: normalized_probs[i] for i in range(4)}
        
        return {
            'label': label,
            'label_id': label_id,
            'confidence': confidence,
            'probabilities': prob_dict,
            'score': score
        }

    def evaluate(self, question: str, answer: str) -> dict:
        """Evaluate the quality of an interview answer.
        
        Args:
            question: The interview question text
            answer: The candidate's answer text
        
        Returns:
            dict with:
            - 'label': str ('Poor', 'Needs Improvement', 'Good', 'Excellent')
            - 'label_id': int (0-3)
            - 'confidence': float (0-1, probability of predicted class)
            - 'probabilities': dict mapping each label to its probability
            - 'score': int (0-100 percentage score)
        """
        # Guard clause for empty or missing input
        if not answer or not answer.strip():
            return {
                'label': 'Poor',
                'label_id': 0,
                'confidence': 0.99,
                'probabilities': {'Poor': 0.99, 'Needs Improvement': 0.01, 'Good': 0.0, 'Excellent': 0.0},
                'score': 0
            }
            
        # Execute fallback mode if model weights are unavailable
        if self.fallback_mode or self.model is None or self.tokenizer is None:
            return self._fallback_evaluate(question, answer)
            
        # Format input with special tokens
        formatted_input = f"[CLS] {question.strip()} [SEP] {answer.strip()} [SEP]"
        
        # Tokenize with padding and truncation
        inputs = self.tokenizer(
            formatted_input,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt',
            add_special_tokens=False
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Model forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs_tensor = F.softmax(logits, dim=-1).squeeze(0)
            
        probs = probs_tensor.cpu().numpy()
        label_id = int(np.argmax(probs))
        confidence = float(probs[label_id])
        label = LABEL_MAP[label_id]
        
        prob_dict = {LABEL_MAP[i]: round(float(probs[i]), 4) for i in range(4)}
        
        # Calculate continuous, nuanced score (0-100) based on probability distribution
        # Class 0: ~20, Class 1: ~52, Class 2: ~78, Class 3: ~96
        expected_score = (
            float(probs[0]) * 20.0 +
            float(probs[1]) * 52.0 +
            float(probs[2]) * 78.0 +
            float(probs[3]) * 96.0
        )
        score = int(round(max(5.0, min(99.0, expected_score))))
        
        return {
            'label': label,
            'label_id': label_id,
            'confidence': round(confidence, 4),
            'probabilities': prob_dict,
            'score': score
        }

    def generate_feedback(self, question: str, answer: str, eval_result: dict) -> dict:
        """Generate structured feedback based on evaluation.
        
        Uses Chain-of-Thought approach:
        Step 1: Identify what the question is asking
        Step 2: Identify what the answer covers
        Step 3: Identify gaps
        Step 4: Provide actionable improvement suggestions
        
        Returns:
            dict with 'strengths', 'improvements', 'suggestion'
        """
        q_text = question.strip()
        a_text = answer.strip()
        q_lower = q_text.lower()
        a_lower = a_text.lower()
        
        label_id = eval_result.get('label_id', 0)
        score = eval_result.get('score', 0)
        
        # -------------------------------------------------------------------------
        # CHAIN-OF-THOUGHT STEP 1: Identify what the question is asking
        # -------------------------------------------------------------------------
        question_intent = "Technical Explanation"
        expected_elements = ["Core definition", "Mechanism / How it works"]
        
        if any(term in q_lower for term in ["difference", "compare", "vs", "versus"]):
            question_intent = "Comparative Analysis"
            expected_elements = ["Direct comparison criteria", "Advantages & trade-offs", "When to use each"]
        elif any(term in q_lower for term in ["tell me about", "describe a time", "situation", "conflict", "handled"]):
            question_intent = "Behavioral / Scenario"
            expected_elements = ["STAR format (Situation, Task, Action, Result)", "Personal contribution", "Quantifiable impact"]
        elif any(term in q_lower for term in ["how does", "how do", "architecture", "internals", "work under the hood"]):
            question_intent = "Architectural / System Mechanics"
            expected_elements = ["Step-by-step workflow", "Internal components", "Memory / concurrency implications"]
        elif any(term in q_lower for term in ["optimize", "complexity", "scale", "performance"]):
            question_intent = "Performance & Optimization"
            expected_elements = ["Time and space complexity", "Bottlenecks", "Optimization trade-offs"]
        elif any(term in q_lower for term in ["what is", "define", "explain", "meaning"]):
            question_intent = "Conceptual Definition"
            expected_elements = ["Clear definition", "Key characteristics", "Real-world application"]
            
        q_keywords = self._extract_keywords(q_text)
        
        # -------------------------------------------------------------------------
        # CHAIN-OF-THOUGHT STEP 2: Identify what the answer covers
        # -------------------------------------------------------------------------
        words = a_text.split()
        word_count = len(words)
        a_keywords = self._extract_keywords(a_text)
        addressed_keywords = q_keywords.intersection(a_keywords)
        
        # Identify structural components in candidate's response
        has_examples = any(p in a_lower for p in ["for example", "e.g.", "such as", "for instance", "in my project", "case in point"])
        has_reasoning = any(p in a_lower for p in ["because", "since", "due to", "ensures", "in order to", "as a result", "consequently"])
        has_tradeoffs = any(p in a_lower for p in ["trade-off", "however", "on the other hand", "limitation", "drawback", "overhead", "disadvantage"])
        has_complexity = any(p in a_lower for p in ["o(1)", "o(n)", "o(log n)", "o(n^2)", "complexity", "constant time", "linear"])
        has_star = any(p in a_lower for p in ["situation", "task", "action", "result", "my role", "outcome", "metric"])
        
        strengths = []
        
        # Formulate recognized strengths
        if len(addressed_keywords) > 0:
            key_sample = ", ".join(list(addressed_keywords)[:3])
            strengths.append(f"Successfully targeted key question concepts ({key_sample}).")
        
        if word_count >= 50:
            strengths.append(f"Provided sufficient technical breadth with a well-developed {word_count}-word response.")
        elif word_count >= 20:
            strengths.append("Provided a direct, focused response to the core prompt.")
            
        if has_examples:
            strengths.append("Supported theoretical claims with concrete, illustrative examples.")
            
        if has_reasoning:
            strengths.append("Articulated the underlying rationale ('why/how') rather than merely defining terminology.")
            
        if has_tradeoffs:
            strengths.append("Demonstrated senior-level engineering maturity by addressing trade-offs and edge cases.")
            
        if has_complexity:
            strengths.append("Quantified computational efficiency with algorithmic time/space complexity notation.")
            
        if not strengths:
            strengths.append("Acknowledged the core question subject.")

        # -------------------------------------------------------------------------
        # CHAIN-OF-THOUGHT STEP 3: Identify gaps
        # -------------------------------------------------------------------------
        unaddressed_keywords = q_keywords - a_keywords
        gaps = []
        
        if word_count < 25:
            gaps.append("Answer is too concise; lacks sufficient depth and technical exposition expected in interviews.")
            
        if unaddressed_keywords and len(unaddressed_keywords) >= 2:
            gap_sample = ", ".join(list(unaddressed_keywords)[:3])
            gaps.append(f"Missed exploring essential domain aspects: {gap_sample}.")
            
        if not has_examples and label_id <= 2:
            gaps.append("Missing concrete real-world use cases, code analogies, or project examples.")
            
        if question_intent == "Comparative Analysis" and not has_tradeoffs:
            gaps.append("Did not contrast trade-offs (pros/cons) or provide criteria for when to choose one over the other.")
            
        if question_intent == "Behavioral / Scenario" and not (has_star or has_reasoning):
            gaps.append("Did not employ the structured STAR method (Situation, Task, Action, Result) with measurable metrics.")
            
        if not has_reasoning and label_id <= 1:
            gaps.append("Remained purely descriptive without explaining internal mechanisms or root causes.")

        # -------------------------------------------------------------------------
        # CHAIN-OF-THOUGHT STEP 4: Provide actionable improvement suggestions
        # -------------------------------------------------------------------------
        improvements = []
        
        if word_count < 30:
            improvements.append("Elaborate further: walk through the definition, internal mechanics, and a brief real-world example.")
        if not has_examples:
            improvements.append("Include an illustrative example: 'For instance, in a production system...'")
        if not has_tradeoffs:
            improvements.append("Highlight trade-offs: discuss performance costs, memory overhead, or scalability limits.")
        if question_intent == "Comparative Analysis" and len(improvements) < 2:
            improvements.append("Structure the comparison across 3 dimensions: performance, complexity, and typical use cases.")
        if not has_complexity and question_intent in ["Technical Explanation", "Performance & Optimization"]:
            improvements.append("Cite time and space complexity (e.g., O(1) average lookup vs O(N) worst-case).")
            
        if not improvements:
            improvements.append("Discuss advanced optimizations, concurrent access patterns, or modern architectural alternatives.")

        # Synthesize overall actionable guidance
        if label_id == 3: # Excellent
            suggestion = (
                f"Outstanding response! You demonstrated thorough mastery of this {question_intent} question. "
                "To polish it even further for executive or staff-level rounds, briefly mention real-world "
                "production failure modes, monitoring metrics, or how you would scale this across distributed nodes."
            )
        elif label_id == 2: # Good
            suggestion = (
                f"Strong technical answer covering the fundamental concepts. To elevate your rating to 'Excellent', "
                f"incorporate {'concrete trade-offs and complexity metrics' if not has_tradeoffs else 'a specific production example'}. "
                "Structuring your answer with clear transitions (e.g., 'First, the mechanism... Second, the trade-off...') "
                "will maximize clarity under pressure."
            )
        elif label_id == 1: # Needs Improvement
            suggestion = (
                f"You have the basic idea, but your response lacks sufficient engineering depth. For a {question_intent} question, "
                f"interviewers look for: {', '.join(expected_elements)}. "
                "Avoid single-sentence answers: clearly define the term, explain how it operates internally, and provide a concrete use-case."
            )
        else: # Poor
            suggestion = (
                "Your response is either too brief, ambiguous, or misses the core premise of the question. "
                f"Take a structured approach: 1) State a precise one-sentence definition of {q_text[:40]}..., "
                "2) Explain the underlying mechanics or why it matters, and 3) Give a practical example."
            )
            
        return {
            'strengths': strengths,
            'improvements': improvements,
            'suggestion': suggestion,
            'cot_steps': {
                'step1_question_intent': question_intent,
                'step2_coverage': f"Identified {len(addressed_keywords)}/{max(1, len(q_keywords))} key concepts; length={word_count} words",
                'step3_gaps': gaps if gaps else ["No major conceptual omissions identified."],
                'step4_recommendations': improvements
            }
        }


# ---------------------------------------------------------------------------------
# SCRIPT DEMO / TEST SUITE ENTRY POINT
# ---------------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 75)
    print("SMARTHIRE: ANSWER EVALUATOR INFERENCE & COT FEEDBACK TEST")
    print("=" * 75)
    
    # Initialize Evaluator (loads fine-tuned model or falls back gracefully)
    evaluator = AnswerEvaluator(model_path='saved_model')
    
    test_cases = [
        {
            "tier": "Target: Excellent (3)",
            "question": "What is the Python Global Interpreter Lock (GIL) and how does it impact multi-threading?",
            "answer": (
                "The Global Interpreter Lock (GIL) is a synchronization mutex in CPython designed to manage thread-safe "
                "reference counting. It restricts bytecode execution to a single thread at any instant, making standard "
                "multithreading ineffective for CPU-bound tasks. To bypass this limitation, developers use the multiprocessing "
                "module, C-extensions, or alternative implementations like PyPy-STM."
            )
        },
        {
            "tier": "Target: Good (2)",
            "question": "Explain the differences between QuickSort and MergeSort. When would you use one over the other?",
            "answer": (
                "QuickSort is an in-place, unstable divide-and-conquer algorithm with average O(n log n) and worst-case O(n^2) "
                "time complexity. MergeSort is a stable divide-and-conquer algorithm that guarantees O(n log n) time but requires "
                "O(n) auxiliary space. QuickSort is preferred for arrays due to cache locality, while MergeSort is preferred for "
                "linked lists and external sorting."
            )
        },
        {
            "tier": "Target: Needs Improvement (1)",
            "question": "How does database indexing work, and what is the trade-off of adding indexes?",
            "answer": (
                "An index is like a book index. It helps find rows faster without scanning the whole table, "
                "but makes inserts a bit slower."
            )
        },
        {
            "tier": "Target: Poor (0)",
            "question": "Compare REST APIs with GraphQL. What are the key architectural differences?",
            "answer": "REST uses URLs and GraphQL is just newer."
        }
    ]
    
    for i, tc in enumerate(test_cases, 1):
        print(f"\n[{i}] TEST CASE - {tc['tier']}")
        print(f"Q: {tc['question']}")
        print(f"A: {tc['answer']}")
        print("-" * 75)
        
        # 1. Run Evaluation
        eval_result = evaluator.evaluate(tc['question'], tc['answer'])
        print(f"Prediction: {eval_result['label']} (Class ID: {eval_result['label_id']})")
        print(f"Confidence: {eval_result['confidence'] * 100:.2f}% | Score: {eval_result['score']}/100")
        print(f"Probabilities: {eval_result['probabilities']}")
        
        # 2. Generate Structured Feedback
        feedback = evaluator.generate_feedback(tc['question'], tc['answer'], eval_result)
        print("\nChain-of-Thought Feedback:")
        print(f"  Strengths:")
        for s in feedback['strengths']:
            print(f"    + {s}")
        print(f"  Areas for Improvement:")
        for imp in feedback['improvements']:
            print(f"    - {imp}")
        print(f"  Actionable Suggestion:\n    {feedback['suggestion']}")
        print("=" * 75)
