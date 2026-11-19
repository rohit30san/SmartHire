"""
Metrics Evaluation Module
=========================
Text Evaluation Metrics for Natural Language Processing (BLEU & ROUGE).

This module implements NLP evaluation metrics for evaluating interview answers,
generated follow-up questions, and candidate responses.

Academic Context / Viva Notes:
------------------------------
Evaluating natural language generation (NLG) is fundamentally different from
classification. There is no single 'correct' string; multiple formulations
express the exact same semantic truth. Two complementary metric families are standard:

1. BLEU (Bilingual Evaluation Understudy - Papineni et al., 2002):
   - Focus: PRECISION
   - Question: "Of the words the candidate generated, how many actually match the reference?"
   - High BLEU means the candidate avoided generating incorrect / irrelevant words.

2. ROUGE (Recall-Oriented Understudy for Gisting Evaluation - Lin, 2004):
   - Focus: RECALL
   - Question: "Of the key concepts in the reference answer, how many did the candidate capture?"
   - High ROUGE means the candidate provided a complete and thorough answer.
"""

from collections import Counter
import math
import re
from typing import Any, Dict, List, Optional, Tuple

# Try importing rouge_score; if not installed, built-in pure Python fallback is used
try:
    from rouge_score import rouge_scorer
    ROUGE_SCORER_AVAILABLE = True
except ImportError:
    ROUGE_SCORER_AVAILABLE = False

# Try importing nltk; provide robust fallback if punkt data is not downloaded
try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


def tokenize(text: str) -> List[str]:
    """Tokenize a string into a list of lowercase word tokens.

    VIVA EXPLANATION:
    -----------------
    Tokenization is the initial stage of any NLP pipeline. Here we convert text
    to lowercase to ensure case-insensitive matching (e.g., 'Python' matches 'python')
    and strip punctuation to isolate pure word units (alphanumerics).
    
    This function uses a regex pattern `\\b\\w+\\b` that handles contractions
    and word boundaries reliably without external download dependencies.
    """
    if not text:
        return []
    # Standardize whitespace and lowercase
    text = text.lower().strip()
    # Extract alphanumeric word tokens
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Extract all n-grams of order n from a token sequence.

    VIVA EXPLANATION:
    -----------------
    An n-gram is a contiguous sequence of n items from a given sample of text:
    - n=1 (Unigram): Individual words -> ['process', 'is', 'isolated']
    - n=2 (Bigram): Word pairs -> [('process', 'is'), ('is', 'isolated')]
    - n=3 (Trigram): Word triplets -> [('process', 'is', 'isolated')]
    - n=4 (4-gram): Phrase units -> captures grammatical structure & fluency
    """
    if len(tokens) < n or n <= 0:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def compute_bleu(reference: str, candidate: str, max_n: int = 4, smoothing: bool = True) -> Dict[str, Any]:
    """Compute BLEU-1 through BLEU-N scores from scratch.

    VIVA EXPLANATION & MATHEMATICAL FORMULATION:
    --------------------------------------------
    BLEU (Bilingual Evaluation Understudy, Papineni et al., 2002) is a precision-based
    metric developed originally for Machine Translation, now widely used for NLG.

    Step 1: Modified N-gram Precision (p_n):
    ----------------------------------------
    Standard precision counts can be 'cheated'. If a candidate repeats a single common
    word ("the the the the"), standard precision would give 100%.
    
    Modified N-gram Precision clips the candidate n-gram counts to the maximum
    count of that n-gram in the reference text:
        Count_clip(ngram) = min(Count_cand(ngram), Count_ref(ngram))
        p_n = sum_{ngram in Cand} Count_clip(ngram) / sum_{ngram in Cand} Count_cand(ngram)

    Step 2: Brevity Penalty (BP):
    -----------------------------
    Precision favors short candidates. A single-word candidate "process" matching the
    reference would score p_1 = 1.0 (100% precision).
    BP penalizes candidates that are shorter than the reference:
        c = length of candidate (tokens)
        r = length of reference (tokens)
        BP = 1.0             if c > r
        BP = exp(1 - r / c)  if c <= r  (with BP = 0 if c == 0)

    Step 3: Cumulative BLEU-N Score:
    --------------------------------
    BLEU-N is the geometric mean of modified precisions up to order N, scaled by BP:
        BLEU-N = BP * exp( sum_{n=1}^N w_n * ln(p_n) )
    where uniform weights w_n = 1 / N are typically applied.

    Smoothing (Chen & Cherry, 2014):
    If any p_n is 0 (common for higher-order n-grams like 4-grams in short sentences),
    ln(0) = -inf, causing BLEU to collapse to 0. Additive epsilon smoothing prevents
    zero collapse while preserving relative ranking.

    Args:
        reference: Ground truth reference text (string)
        candidate: Model or student generated text (string)
        max_n: Maximum n-gram order (default 4)
        smoothing: Whether to apply epsilon smoothing for zero-count n-grams

    Returns:
        Dictionary containing:
        - 'bleu_1' through 'bleu_N': Cumulative BLEU scores (0.0 to 1.0)
        - 'brevity_penalty': BP factor (0.0 to 1.0)
        - 'precisions': List of individual modified precisions [p_1, ..., p_N]
        - 'candidate_len': Number of tokens in candidate
        - 'reference_len': Number of tokens in reference
    """
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)

    c = len(cand_tokens)
    r = len(ref_tokens)

    # Edge cases: empty candidate or reference
    if c == 0:
        return {
            f'bleu_{i}': 0.0 for i in range(1, max_n + 1)
        } | {
            'brevity_penalty': 0.0,
            'precisions': [0.0] * max_n,
            'candidate_len': 0,
            'reference_len': r
        }

    # Step 1: Compute Brevity Penalty (BP)
    if c > r:
        bp = 1.0
    else:
        # Penalize short answers exponentially
        bp = math.exp(1.0 - (float(r) / float(c))) if c > 0 else 0.0

    # Step 2: Calculate Modified N-gram Precisions (p_1 to p_N)
    precisions: List[float] = []

    for n in range(1, max_n + 1):
        cand_ngrams = get_ngrams(cand_tokens, n)
        ref_ngrams = get_ngrams(ref_tokens, n)

        if not cand_ngrams:
            # Candidate is shorter than n-gram order
            precisions.append(0.0)
            continue

        cand_counts = Counter(cand_ngrams)
        ref_counts = Counter(ref_ngrams)

        # Clip each candidate n-gram count to maximum reference count
        clipped_matches = 0
        for ngram, count in cand_counts.items():
            clipped_matches += min(count, ref_counts.get(ngram, 0))

        total_cand_ngrams = len(cand_ngrams)
        p_n = float(clipped_matches) / float(total_cand_ngrams)

        precisions.append(p_n)

    # Step 3: Compute Cumulative BLEU-1 through BLEU-N
    results: Dict[str, Any] = {
        'brevity_penalty': round(bp, 4),
        'precisions': [round(p, 4) for p in precisions],
        'candidate_len': c,
        'reference_len': r
    }

    epsilon = 1e-9  # Smoothing factor to prevent log(0) = -inf

    for k in range(1, max_n + 1):
        # Uniform weights for cumulative BLEU-k: w_i = 1 / k
        weights = [1.0 / k] * k
        k_precisions = precisions[:k]

        # Calculate geometric mean using log-probabilities
        log_precision_sum = 0.0
        has_zero = False

        for p, w in zip(k_precisions, weights):
            if p > 0:
                log_precision_sum += w * math.log(p)
            else:
                has_zero = True
                if smoothing:
                    log_precision_sum += w * math.log(epsilon)
                else:
                    break

        if has_zero and not smoothing:
            bleu_score = 0.0
        else:
            bleu_score = bp * math.exp(log_precision_sum)

        # Clamp between 0.0 and 1.0
        bleu_score = max(0.0, min(1.0, bleu_score))
        results[f'bleu_{k}'] = round(bleu_score, 4)

    return results


def _compute_lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """Compute the Longest Common Subsequence (LCS) length via 2D Dynamic Programming.

    VIVA EXPLANATION:
    -----------------
    LCS finds the longest sequence of words that appears in both texts in the
    SAME ORDER, but NOT necessarily contiguously.
    
    Example:
    Seq 1: ['a', 'process', 'runs', 'isolated']
    Seq 2: ['a', 'process', 'is', 'completely', 'isolated']
    LCS: ['a', 'process', 'isolated'] -> Length = 3
    
    Contrast with Substring: A substring requires consecutive words; a subsequence
    only requires preserving relative order. This gives ROUGE-L flexibility to match
    natural phrasing variations!
    
    Time Complexity: O(m * n) where m, n are token lengths.
    Space Complexity: O(m * n) DP matrix.
    """
    m, n = len(seq1), len(seq2)
    # Initialize DP table of size (m+1) x (n+1)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def _compute_rouge_fallback(reference: str, candidate: str) -> Dict[str, Dict[str, float]]:
    """Pure-Python implementation of ROUGE-1, ROUGE-2, and ROUGE-L.

    Serves as an educational reference and fallback when rouge_score is not installed.
    """
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)

    def _calc_prf(matches: int, cand_total: int, ref_total: int) -> Dict[str, float]:
        precision = float(matches) / float(cand_total) if cand_total > 0 else 0.0
        recall = float(matches) / float(ref_total) if ref_total > 0 else 0.0
        if precision + recall > 0:
            fmeasure = (2.0 * precision * recall) / (precision + recall)
        else:
            fmeasure = 0.0
        return {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'fmeasure': round(fmeasure, 4)
        }

    # ROUGE-1 (Unigrams)
    cand_unigrams = Counter(cand_tokens)
    ref_unigrams = Counter(ref_tokens)
    r1_matches = sum(min(count, ref_unigrams.get(tok, 0)) for tok, count in cand_unigrams.items())
    rouge_1 = _calc_prf(r1_matches, len(cand_tokens), len(ref_tokens))

    # ROUGE-2 (Bigrams)
    cand_bigrams = Counter(get_ngrams(cand_tokens, 2))
    ref_bigrams = Counter(get_ngrams(ref_tokens, 2))
    r2_matches = sum(min(count, ref_bigrams.get(bg, 0)) for bg, count in cand_bigrams.items())
    rouge_2 = _calc_prf(r2_matches, len(cand_tokens) - 1, len(ref_tokens) - 1)

    # ROUGE-L (Longest Common Subsequence)
    lcs_len = _compute_lcs_length(cand_tokens, ref_tokens)
    rouge_l = _calc_prf(lcs_len, len(cand_tokens), len(ref_tokens))

    return {
        'rouge_1': rouge_1,
        'rouge_2': rouge_2,
        'rouge_l': rouge_l
    }


def compute_rouge(reference: str, candidate: str) -> Dict[str, Dict[str, float]]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.

    VIVA EXPLANATION:
    -----------------
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation - Chin-Yew Lin, 2004):
    While BLEU prioritizes Precision, ROUGE prioritizes RECALL:
    'Of all the critical concepts present in the reference answer, how many
    did the candidate actually include?'

    Metrics explained:
    - ROUGE-1: Unigram recall, precision, and F1. Measures coverage of individual key terms.
    - ROUGE-2: Bigram recall, precision, and F1. Measures capture of 2-word technical phrases.
    - ROUGE-L: Longest Common Subsequence (LCS).
      Takes sentence structure into account without requiring exact consecutive phrase matching.
      It rewards words appearing in the correct sequence across the sentence.

    Key Formulas for Viva:
    - Recall = Matches / Reference_Length
    - Precision = Matches / Candidate_Length
    - F1-Measure = 2 * (Precision * Recall) / (Precision + Recall) (Harmonic Mean)

    Returns:
        dict with keys: 'rouge_1', 'rouge_2', 'rouge_l'
        Each value is a dict with: 'precision', 'recall', 'fmeasure'
    """
    if not candidate.strip() or not reference.strip():
        empty_score = {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}
        return {
            'rouge_1': empty_score.copy(),
            'rouge_2': empty_score.copy(),
            'rouge_l': empty_score.copy()
        }

    # Use standard Hugging Face / Google rouge_scorer if available
    if ROUGE_SCORER_AVAILABLE:
        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(reference, candidate)
            return {
                'rouge_1': {
                    'precision': round(scores['rouge1'].precision, 4),
                    'recall': round(scores['rouge1'].recall, 4),
                    'fmeasure': round(scores['rouge1'].fmeasure, 4)
                },
                'rouge_2': {
                    'precision': round(scores['rouge2'].precision, 4),
                    'recall': round(scores['rouge2'].recall, 4),
                    'fmeasure': round(scores['rouge2'].fmeasure, 4)
                },
                'rouge_l': {
                    'precision': round(scores['rougeL'].precision, 4),
                    'recall': round(scores['rougeL'].recall, 4),
                    'fmeasure': round(scores['rougeL'].fmeasure, 4)
                }
            }
        except Exception:
            # Fall back to pure-Python implementation if rouge_scorer fails
            return _compute_rouge_fallback(reference, candidate)
    else:
        # Pure-Python fallback implementation
        return _compute_rouge_fallback(reference, candidate)


def compute_all_metrics(reference: str, candidate: str, max_n: int = 4) -> Dict[str, Any]:
    """Compute both BLEU and ROUGE metrics and return a unified results dictionary.

    VIVA EXPLANATION:
    -----------------
    In an AI interview evaluation platform like SmartHire, neither BLEU nor ROUGE
    is sufficient in isolation:
    - Candidate with high BLEU but low ROUGE:
      Gave a brief, accurate statement, but missed majority of the technical answer.
    - Candidate with low BLEU but high ROUGE:
      Rambled with many unnecessary words (diluting precision), but eventually
      touched on all the reference concepts (high recall).
    - High BLEU AND High ROUGE:
      The ideal technical answer: accurate, concise, and comprehensive.

    Args:
        reference: Reference answer / ground truth string
        candidate: Candidate's response string
        max_n: Maximum n-gram order for BLEU (default 4)

    Returns:
        Comprehensive dictionary containing reference, candidate, bleu, and rouge metrics.
    """
    bleu_results = compute_bleu(reference, candidate, max_n=max_n)
    rouge_results = compute_rouge(reference, candidate)

    return {
        'reference': reference,
        'candidate': candidate,
        'token_counts': {
            'reference_tokens': bleu_results['reference_len'],
            'candidate_tokens': bleu_results['candidate_len'],
        },
        'bleu': bleu_results,
        'rouge': rouge_results,
    }


def format_metrics_report(metrics_dict: Dict[str, Any]) -> str:
    """Format metrics dictionary into a clean, human-readable terminal report.

    Args:
        metrics_dict: Dictionary returned by compute_all_metrics()

    Returns:
        Formatted ASCII report string suitable for display or logging.
    """
    ref = metrics_dict.get('reference', '')
    cand = metrics_dict.get('candidate', '')
    bleu = metrics_dict.get('bleu', {})
    rouge = metrics_dict.get('rouge', {})
    tokens = metrics_dict.get('token_counts', {})

    report_lines = [
        "=" * 78,
        "SMARTHIRE NLP EVALUATION METRICS REPORT",
        "=" * 78,
        f"REFERENCE TEXT : \"{ref}\"",
        f"CANDIDATE TEXT : \"{cand}\"",
        "-" * 78,
        f"Token Count    : Candidate = {tokens.get('candidate_tokens', 'N/A')} words | "
        f"Reference = {tokens.get('reference_tokens', 'N/A')} words",
        f"Brevity Penalty: {bleu.get('brevity_penalty', 0.0):.4f} "
        f"({'No penalty (cand >= ref)' if bleu.get('brevity_penalty', 0.0) == 1.0 else 'Short candidate penalized'})",
        "-" * 78,
        "1. BLEU SCORES (Precision Oriented)",
        "   Measures: What fraction of candidate n-grams appear in the reference?",
        f"   - BLEU-1 (Unigram Precision) : {bleu.get('bleu_1', 0.0):.4f}",
        f"   - BLEU-2 (Bigram Precision)  : {bleu.get('bleu_2', 0.0):.4f}",
        f"   - BLEU-3 (Trigram Precision) : {bleu.get('bleu_3', 0.0):.4f}",
        f"   - BLEU-4 (4-gram Precision)  : {bleu.get('bleu_4', 0.0):.4f}",
        f"   - Individual Precisions (p_n): {bleu.get('precisions', [])}",
        "-" * 78,
        "2. ROUGE SCORES (Recall Oriented)",
        "   Measures: What fraction of reference content was captured by the candidate?",
        f"   - ROUGE-1 : Precision={rouge.get('rouge_1', {}).get('precision', 0.0):.4f} | "
        f"Recall={rouge.get('rouge_1', {}).get('recall', 0.0):.4f} | "
        f"F1={rouge.get('rouge_1', {}).get('fmeasure', 0.0):.4f}",
        f"   - ROUGE-2 : Precision={rouge.get('rouge_2', {}).get('precision', 0.0):.4f} | "
        f"Recall={rouge.get('rouge_2', {}).get('recall', 0.0):.4f} | "
        f"F1={rouge.get('rouge_2', {}).get('fmeasure', 0.0):.4f}",
        f"   - ROUGE-L : Precision={rouge.get('rouge_l', {}).get('precision', 0.0):.4f} | "
        f"Recall={rouge.get('rouge_l', {}).get('recall', 0.0):.4f} | "
        f"F1={rouge.get('rouge_l', {}).get('fmeasure', 0.0):.4f}",
        "-" * 78,
        "3. VIVA EVALUATION SUMMARY",
    ]

    r1_f1 = rouge.get('rouge_1', {}).get('fmeasure', 0.0)
    bleu_2 = bleu.get('bleu_2', 0.0)

    if r1_f1 > 0.6 and bleu_2 > 0.4:
        summary = "STRONG ANSWER: High precision and comprehensive concept recall."
    elif r1_f1 > 0.5 and bleu_2 <= 0.4:
        summary = "ACCEPTABLE BUT VERBOSE: Captured main concepts but word choice/phrasing drifted."
    elif r1_f1 <= 0.5 and bleu.get('brevity_penalty', 1.0) < 0.6:
        summary = "INCOMPLETE ANSWER: Penalized for excessive brevity; missed key technical points."
    else:
        summary = "WEAK ANSWER: Low overlap with reference terminology and concepts."

    report_lines.append(f"   Assessment: {summary}")
    report_lines.append("=" * 78)

    return "\n".join(report_lines)


# =====================================================================
# Standalone Demonstration & Viva Explanation Execution Block
# =====================================================================
if __name__ == '__main__':
    print("=" * 78)
    print("SMARTHIRE: Natural Language Generation Evaluation Demonstration")
    print("Lab Evaluation: BLEU (Precision) vs ROUGE (Recall) in Interview Scoring")
    print("=" * 78)

    # Reference Ground Truth Answer
    reference_answer = (
        "A process is an executing program instance with its own private address space "
        "and isolated memory, while a thread is a lightweight execution unit within a "
        "process that shares the process memory and resources."
    )

    # Scenario A: Excellent Candidate Answer (Concise & Accurate)
    candidate_good = (
        "A process is a running program with private memory space, whereas a thread is a "
        "lightweight unit of execution that shares memory with its parent process."
    )

    # Scenario B: Extremely Brief Answer (Suffers from Brevity Penalty & Low Recall)
    candidate_short = (
        "A process has memory and threads are inside processes."
    )

    # Scenario C: Rambling / Verbose Answer (Diluted Precision)
    candidate_verbose = (
        "In modern operating systems like Linux and Windows, computer programs have "
        "many things happening. Processes exist for executing applications, and they have "
        "some memory, and then developers use multithreading to speed up CPU usage."
    )

    scenarios = [
        ("Candidate A (Strong Technical Answer)", candidate_good),
        ("Candidate B (Excessively Brief Answer)", candidate_short),
        ("Candidate C (Verbose / Vague Answer)", candidate_verbose),
    ]

    for label, candidate_text in scenarios:
        print(f"\nEvaluating: {label}")
        results = compute_all_metrics(reference_answer, candidate_text)
        report = format_metrics_report(results)
        print(report)

    # Key Viva Takeaway Summary Table
    print("\n" + "=" * 78)
    print("VIVA TAKEAWAYS: KEY DIFFERENCES BETWEEN BLEU AND ROUGE")
    print("=" * 78)
    print("Feature                 | BLEU                              | ROUGE")
    print("-" * 78)
    print("Primary Focus           | Precision                         | Recall")
    print("Core Question           | 'Are generated words accurate?'  | 'Are reference ideas covered?'")
    print("Penalty Mechanism       | Brevity Penalty (for short text)  | N/A (inherently penalizes brevity)")
    print("Structure Alignment    | N-gram matches (1 to 4)           | Longest Common Subsequence (LCS)")
    print("Originating Domain      | Machine Translation (Papineni 02) | Text Summarization (Lin 2004)")
    print("Best Used When          | Ensuring factual conciseness      | Ensuring complete technical breadth")
    print("=" * 78)
