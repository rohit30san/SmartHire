"""
prompts.py - Prompt Engineering Templates for SmartHire

This module demonstrates three key prompt engineering techniques
from the Generative AI syllabus (Unit 5):

1. Interview Pattern Approach:
   - Structures the AI as an interviewer persona
   - Maintains multi-turn conversational context
   - Adapts follow-up questions based on responses

2. Chain-of-Thought (CoT) Approach:
   - Forces step-by-step reasoning before final answer
   - Used for evaluating answer quality systematically
   - Reduces errors by decomposing complex judgments

3. Tree-of-Thought (ToT) Approach:
   - Explores multiple reasoning paths before converging
   - Used for generating diverse interview questions
   - Evaluates multiple hypotheses before selecting the best
"""

import re
import string
from typing import Dict, List, Tuple, Set, Any, Optional, Union

# ============================================================
# 1. INTERVIEW PATTERN APPROACH
# ============================================================
# VIVA EXPLANATION:
# The Interview Pattern constrains the LLM to behave as a
# structured interviewer. It maintains persona, controls
# question flow, and adapts based on candidate responses.
# This is different from open-ended generation because the
# output must follow a specific conversational structure:
# - One question at a time (preventing prompt overwhelm).
# - Persona conditioning: prevents breaking character or hallucinating answers.
# - Multi-turn state tracking: passing conversation transcripts into context.

INTERVIEW_SYSTEM_PROMPT = """
You are a professional technical interviewer conducting a {difficulty} level 
interview for a {role} position. Follow these rules strictly:

1. Ask ONE question at a time
2. Keep questions focused on {role} skills
3. Adjust complexity based on the candidate's previous answers
4. Be professional and encouraging
5. After each answer, briefly acknowledge before asking the next question

Difficulty guidelines:
- Easy: Fundamental concepts, definitions, basic syntax
- Medium: Applied knowledge, problem-solving, design decisions  
- Hard: System design, optimization, edge cases, trade-offs
"""

# Template for building the generation prompt with conversation history
INTERVIEW_CONTEXT_TEMPLATE = """
{system_prompt}

Interview transcript so far:
{conversation_history}

Based on the candidate's last answer, ask the next relevant interview question.
Interviewer:"""


# ============================================================
# 2. CHAIN-OF-THOUGHT (CoT) APPROACH  
# ============================================================
# VIVA EXPLANATION:
# Chain-of-Thought prompting (Wei et al., 2022) instructs the model to show its
# reasoning step-by-step before giving a final answer.
# Research shows this dramatically improves accuracy on reasoning tasks.
# Instead of jumping to an arbitrary score, CoT decomposes evaluation into:
# 1. Relevance: Did they answer the question asked?
# 2. Concept Coverage: Did they hit the essential technical keywords?
# 3. Depth: Is the explanation shallow or comprehensive?
# 4. Structure: Is the communication logical and professional?
# 5. Examples/Nuance: Did they mention edge cases, syntax, or trade-offs?

COT_EVALUATION_TEMPLATE = """
Evaluate the following interview answer step by step.

Question: {question}
Candidate's Answer: {answer}
Expected Key Concepts: {key_concepts}

Let's think step by step:

Step 1 - Relevance Check: Is the answer relevant to the question asked?
{relevance_analysis}

Step 2 - Concept Coverage: Which key concepts did the candidate mention?
{concept_analysis}

Step 3 - Depth Assessment: How detailed and specific is the answer?
{depth_analysis}

Step 4 - Communication Quality: Is the answer well-structured and clear?
{communication_analysis}

Final Evaluation: Based on the above analysis...
Quality Rating: {rating}
Score: {score}/100
Key Feedback: {feedback}
"""

# Simplified CoT for rule-based evaluation (no LLM needed)
COT_STEPS = [
    'relevance',    # Is the answer about the right topic?
    'coverage',     # Does it mention the key concepts?
    'depth',        # How detailed is the explanation?
    'structure',    # Is it well-organized?
    'examples'      # Does it include examples or edge cases?
]


# ============================================================
# 3. TREE-OF-THOUGHT (ToT) APPROACH
# ============================================================  
# VIVA EXPLANATION:
# Tree-of-Thought (ToT, Yao et al., 2023) extends CoT by exploring MULTIPLE
# reasoning paths simultaneously, evaluating candidate thoughts, and then
# selecting or backtracking to the optimal path.
# Think of it as:
# - Standard Prompting = Single shot sample (Input -> Output)
# - Chain-of-Thought (CoT) = Linear path (Input -> Step 1 -> Step 2 -> Output)
# - Tree-of-Thought (ToT) = Branching tree of decisions:
#       Input -> [Branch A, Branch B, Branch C] -> Evaluation -> Best Output
#
# In SmartHire, we use ToT for adaptive question selection:
# - Branch A: Probe Weakness (target topic where candidate scored low)
# - Branch B: Deepen Strength (harder question where candidate excelled)
# - Branch C: New Topic (explore untested areas for syllabus coverage)
# An interview policy evaluator then scores each branch to select the best next move.

TOT_QUESTION_GENERATION_TEMPLATE = """
Based on the interview so far, generate three possible next questions:

Candidate Profile:
- Role: {role}
- Performance so far: {performance_summary}
- Topics covered: {topics_covered}
- Topics remaining: {topics_remaining}

Branch A (Probe Weakness): A question that tests an area where the 
candidate showed weakness or uncertainty.
Question A: {branch_a}

Branch B (Deepen Strength): A harder question in an area where the 
candidate showed strong knowledge.
Question B: {branch_b}

Branch C (New Topic): A question on a completely new topic not yet covered.
Question C: {branch_c}

Selected Question (best for current interview stage): {selected}
Reasoning: {reasoning}
"""


# ============================================================
# COMMON STOPWORDS FOR TEXT ANALYSIS
# ============================================================
STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
    'any', 'are', 'aren', 'arent', 'as', 'at', 'be', 'because', 'been', 'before',
    'being', 'below', 'between', 'both', 'but', 'by', 'can', 'cant', 'cannot',
    'could', 'couldnt', 'did', 'didn', 'didnt', 'do', 'does', 'doesn', 'doesnt',
    'doing', 'don', 'dont', 'down', 'during', 'each', 'explain', 'few', 'for',
    'from', 'further', 'had', 'hadn', 'hadnt', 'has', 'hasn', 'hasnt', 'have',
    'haven', 'havent', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him',
    'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', 'isnt', 'it',
    'its', 'itself', 'just', 'me', 'more', 'most', 'must', 'my', 'myself', 'no',
    'nor', 'not', 'now', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our',
    'ours', 'ourselves', 'out', 'over', 'own', 'same', 'should', 'so', 'some',
    'such', 'tell', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves',
    'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too',
    'under', 'until', 'up', 'very', 'was', 'wasn', 'wasnt', 'we', 'were', 'weren',
    'werent', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why',
    'with', 'would', 'you', 'your', 'yours', 'yourself', 'yourselves'
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def format_conversation_history(history: Optional[List[Any]]) -> str:
    """Format conversation history list into a readable transcript string.
    
    Handles lists of (role, text) tuples or list of dicts with 'role' and 'text'/'content'.
    
    Args:
        history: List of (role, text) tuples or dicts representing turns.
        
    Returns:
        Formatted multi-turn transcript string.
    """
    if not history:
        return "No prior conversation recorded."
    
    formatted_turns = []
    for item in history:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            speaker, utterance = item[0], item[1]
        elif isinstance(item, dict):
            speaker = item.get("role", item.get("speaker", "Interviewer"))
            utterance = item.get("text", item.get("content", ""))
        else:
            speaker, utterance = "Speaker", str(item)
            
        speaker_name = str(speaker).strip().capitalize()
        utterance_text = str(utterance).strip()
        formatted_turns.append(f"{speaker_name}: {utterance_text}")
        
    return "\n".join(formatted_turns)


def build_interview_prompt(role: str, difficulty: str, conversation_history: Optional[List[Any]] = None) -> str:
    """Build a complete interview prompt using the Interview Pattern.
    
    VIVA EXPLANATION:
    The Interview Pattern combines persona constraints (system prompt) with dynamic
    multi-turn conversational memory. This ensures the model maintains interviewer
    etiquette, focuses exclusively on role-specific competencies, and smoothly
    references the candidate's last answer.
    
    Args:
        role: Job role (e.g., 'python_developer', 'data_scientist')
        difficulty: 'easy', 'medium', or 'hard'
        conversation_history: List of (role, text) tuples or dicts
    
    Returns:
        Complete formatted prompt string ready for LLM consumption.
    """
    clean_role = role.replace("_", " ").strip().title()
    clean_diff = difficulty.strip().lower()
    if clean_diff not in {"easy", "medium", "hard"}:
        clean_diff = "medium"

    formatted_system = INTERVIEW_SYSTEM_PROMPT.format(
        role=clean_role,
        difficulty=clean_diff
    ).strip()

    formatted_history = format_conversation_history(conversation_history)

    prompt = INTERVIEW_CONTEXT_TEMPLATE.format(
        system_prompt=formatted_system,
        conversation_history=formatted_history
    ).strip()

    return prompt


def get_feedback_for_score(score: int, question: str, answer: str) -> str:
    """Generate tailored feedback based on score range and answer attributes.
    
    VIVA EXPLANATION:
    Feedback generation maps quantitative evaluation back into qualitative,
    actionable coaching. In Unit 5, automated feedback loops help students
    understand knowledge gaps without human grading bottlenecks.
    
    Args:
        score: Integer score from 0 to 100
        question: The interview question asked
        answer: The candidate's response
        
    Returns:
        str: Constructive feedback with specific suggestions for improvement.
    """
    if score >= 85:
        feedback = (
            "Excellent answer! You demonstrated a comprehensive, nuanced understanding "
            "of the core principles, accurately incorporating key technical vocabulary "
            "and practical reasoning. To further stand out, consider discussing system-level "
            "trade-offs, concurrency aspects, or edge-case handling."
        )
    elif score >= 70:
        feedback = (
            "Good response. You captured the primary concept and answered accurately. "
            "To elevate this response to an advanced level, provide concrete code syntax, "
            "elaborate on performance characteristics (e.g., memory/time complexity), "
            "or explain how this mechanism operates internally under the hood."
        )
    elif score >= 50:
        feedback = (
            "Needs Improvement. Your answer touched upon the relevant subject area but was "
            "either too brief or omitted essential technical concepts. Focus on defining "
            "fundamental terminology explicitly and explaining the 'why' and 'how' rather "
            "than just making high-level statements."
        )
    else:
        feedback = (
            "Poor response. The answer either deviated significantly from the question asked "
            "or lacked basic technical accuracy and depth. We recommend revisiting the foundational "
            "principles of this topic, practicing formal definitions, and reviewing standard examples."
        )
        
    return feedback


def _extract_keywords(text: str) -> Set[str]:
    """Helper to extract non-stopword alphanumeric tokens from text."""
    clean_text = text.lower()
    tokens = re.findall(r'\b[a-z0-9_]{3,}\b', clean_text)
    return {token for token in tokens if token not in STOPWORDS}


def build_cot_evaluation(question: str, answer: str, key_concepts: List[str]) -> Dict[str, Any]:
    """Build a Chain-of-Thought evaluation for an answer.
    
    VIVA EXPLANATION:
    Chain-of-Thought (CoT) decomposes a complex judgment into transparent,
    verifiable steps. Instead of producing an arbitrary score in one step,
    this function performs rule-based evaluation through explicit cognitive steps:
    1. Relevance Check: Measures keyword alignment with the question (Max 20 pts).
    2. Concept Coverage: Checks presence of expected technical keywords (Max 35 pts).
    3. Depth Assessment: Analyzes word count and technical specificity (Max 20 pts).
    4. Structure & Communication: Evaluates sentence formation & cohesion (Max 15 pts).
    5. Examples & Practical Nuance: Detects practical markers and edge-case awareness (Max 10 pts).
    
    Args:
        question: The interview question
        answer: Candidate's response text
        key_concepts: List of expected key concepts/terms (e.g., ['yield', 'iterator'])
        
    Returns:
        dict with:
        - 'steps': list of step-by-step analysis strings
        - 'rating': 'Poor'/'Needs Improvement'/'Good'/'Excellent'
        - 'score': int 0-100
        - 'feedback': str with improvement suggestions
        - 'breakdown': dict with individual step scores
        - 'formatted_cot': str with fully formatted COT_EVALUATION_TEMPLATE
    """
    answer_clean = answer.strip()
    answer_lower = answer_clean.lower()
    
    # ----------------------------------------------------
    # Step 1: Relevance Check (0 - 20 points)
    # ----------------------------------------------------
    q_keywords = _extract_keywords(question)
    a_keywords = _extract_keywords(answer)
    
    overlap = q_keywords.intersection(a_keywords)
    if not q_keywords:
        relevance_ratio = 1.0
    else:
        # Ratio of question keywords covered or relevant answer overlap
        relevance_ratio = len(overlap) / max(len(q_keywords), 1)
        
    # Scale relevance: minimum threshold if candidate writes a plausible response
    if len(answer_clean.split()) < 3:
        score_relevance = 2
        rel_comment = f"Answer is extremely brief ({len(answer_clean.split())} words) with minimal relevance."
    elif relevance_ratio >= 0.4 or len(overlap) >= 2:
        score_relevance = 20
        rel_comment = f"High relevance. Matched key question anchors: {sorted(list(overlap))}."
    elif len(overlap) == 1:
        score_relevance = 14
        rel_comment = f"Moderate relevance. Touched upon key term '{list(overlap)[0]}' but strayed somewhat."
    else:
        score_relevance = 8
        rel_comment = "Low keyword overlap with question. Answer appears generic or tangential."

    # ----------------------------------------------------
    # Step 2: Concept Coverage (0 - 35 points)
    # ----------------------------------------------------
    matched_concepts = []
    missing_concepts = []
    
    for concept in key_concepts:
        concept_clean = concept.strip().lower()
        # Direct substring match or token-level match
        if concept_clean in answer_lower:
            matched_concepts.append(concept)
        else:
            # Check if all words of a multi-word concept appear
            concept_words = _extract_keywords(concept)
            if concept_words and concept_words.issubset(a_keywords):
                matched_concepts.append(concept)
            else:
                missing_concepts.append(concept)

    total_expected = len(key_concepts)
    if total_expected > 0:
        coverage_ratio = len(matched_concepts) / total_expected
        score_coverage = round(coverage_ratio * 35)
        cov_comment = (
            f"Covered {len(matched_concepts)}/{total_expected} expected concepts. "
            f"Identified: {matched_concepts if matched_concepts else 'None'}. "
            f"Missing: {missing_concepts if missing_concepts else 'None'}."
        )
    else:
        score_coverage = 30
        cov_comment = "No specific concept keywords supplied; assessed based on general technical vocabulary."

    # ----------------------------------------------------
    # Step 3: Depth Assessment (0 - 20 points)
    # ----------------------------------------------------
    words = answer_clean.split()
    word_count = len(words)
    
    if word_count < 8:
        score_depth = 4
        depth_comment = f"Shallow explanation ({word_count} words). Lacks technical elaboration."
    elif word_count < 20:
        score_depth = 10
        depth_comment = f"Brief explanation ({word_count} words). Covers basic surface level."
    elif word_count < 45:
        score_depth = 16
        depth_comment = f"Moderate depth ({word_count} words). Provides clear explanation with context."
    else:
        score_depth = 20
        depth_comment = f"Comprehensive depth ({word_count} words). Detailed and technically descriptive."

    # ----------------------------------------------------
    # Step 4: Communication Quality & Structure (0 - 15 points)
    # ----------------------------------------------------
    # Count sentences
    sentences = [s.strip() for s in re.split(r'[.!?]+', answer_clean) if len(s.strip()) > 0]
    sentence_count = len(sentences)
    
    # Discourse connectors
    connectors = {'because', 'however', 'therefore', 'furthermore', 'for example', 
                  'specifically', 'whereas', 'additionally', 'in contrast', 'first', 'second'}
    connector_found = [c for c in connectors if c in answer_lower]
    
    if sentence_count <= 1 and word_count < 15:
        score_structure = 6
        comm_comment = "Fragmented single-sentence response without connective discourse."
    elif sentence_count >= 2 and connector_found:
        score_structure = 15
        comm_comment = (
            f"Well-structured across {sentence_count} sentences. Effective use of "
            f"connectors ({connector_found[:2]}) and logical organization."
        )
    elif sentence_count >= 2:
        score_structure = 12
        comm_comment = f"Good coherent flow divided into {sentence_count} logical sentences."
    else:
        score_structure = 9
        comm_comment = "Acceptable sentence structure, but could benefit from clearer multi-point breakdown."

    # ----------------------------------------------------
    # Step 5: Examples & Practical Nuance (0 - 10 points)
    # ----------------------------------------------------
    example_patterns = [
        r'\bfor example\b', r'\be\.g\.\b', r'\bsuch as\b', r'\bfor instance\b',
        r'\bdef \b', r'\breturn\b', r'\bclass \b', r'\bimport \b', r'\bo\([1nlog\^]+\)',
        r'\btrade-off\b', r'\boverhead\b', r'\bmemory leak\b', r'\bperformance\b'
    ]
    examples_found = [p for p in example_patterns if re.search(p, answer_lower)]
    
    if len(examples_found) >= 2:
        score_examples = 10
        ex_comment = "Includes practical code keywords, complexity references, or concrete examples."
    elif len(examples_found) == 1:
        score_examples = 6
        ex_comment = "Mentions practical example or syntax marker."
    else:
        score_examples = 2
        ex_comment = "Theoretical only; lacks concrete examples, syntax cues, or edge cases."

    # ----------------------------------------------------
    # Final Aggregation
    # ----------------------------------------------------
    total_score = score_relevance + score_coverage + score_depth + score_structure + score_examples
    total_score = max(0, min(100, total_score))

    if total_score >= 85:
        rating = "Excellent"
    elif total_score >= 70:
        rating = "Good"
    elif total_score >= 50:
        rating = "Needs Improvement"
    else:
        rating = "Poor"

    feedback = get_feedback_for_score(total_score, question, answer)
    if missing_concepts and rating != "Excellent":
        feedback += f" Specific concepts to review: {', '.join(missing_concepts)}."

    steps = [
        f"Step 1 - Relevance Check (Weight: 20, Score: {score_relevance}/20): {rel_comment}",
        f"Step 2 - Concept Coverage (Weight: 35, Score: {score_coverage}/35): {cov_comment}",
        f"Step 3 - Depth Assessment (Weight: 20, Score: {score_depth}/20): {depth_comment}",
        f"Step 4 - Communication Quality (Weight: 15, Score: {score_structure}/15): {comm_comment}",
        f"Step 5 - Examples & Practical Nuance (Weight: 10, Score: {score_examples}/10): {ex_comment}"
    ]

    formatted_cot = COT_EVALUATION_TEMPLATE.format(
        question=question.strip(),
        answer=answer_clean,
        key_concepts=", ".join(key_concepts) if key_concepts else "None specified",
        relevance_analysis=f"Score {score_relevance}/20: {rel_comment}",
        concept_analysis=f"Score {score_coverage}/35: {cov_comment}",
        depth_analysis=f"Score {score_depth}/20: {depth_comment} (Examples: {score_examples}/10)",
        communication_analysis=f"Score {score_structure}/15: {comm_comment}",
        rating=rating,
        score=total_score,
        feedback=feedback
    ).strip()

    return {
        "steps": steps,
        "rating": rating,
        "score": total_score,
        "feedback": feedback,
        "breakdown": {
            "relevance": score_relevance,
            "coverage": score_coverage,
            "depth": score_depth,
            "structure": score_structure,
            "examples": score_examples
        },
        "formatted_cot": formatted_cot
    }


def _normalize_question_bank(question_bank: Any) -> List[Dict[str, Any]]:
    """Helper to convert various question bank representations into a unified list."""
    normalized: List[Dict[str, Any]] = []
    
    if isinstance(question_bank, dict):
        for q_id, q_data in question_bank.items():
            if isinstance(q_data, dict):
                item = dict(q_data)
                if "id" not in item:
                    item["id"] = q_id
                normalized.append(item)
            elif isinstance(q_data, str):
                normalized.append({
                    "id": q_id,
                    "question": q_data,
                    "topic": "general",
                    "difficulty": "medium",
                    "key_concepts": []
                })
    elif isinstance(question_bank, (list, tuple)):
        for idx, q_data in enumerate(question_bank):
            if isinstance(q_data, dict):
                item = dict(q_data)
                if "id" not in item:
                    item["id"] = idx
                normalized.append(item)
            elif isinstance(q_data, str):
                normalized.append({
                    "id": idx,
                    "question": q_data,
                    "topic": "general",
                    "difficulty": "medium",
                    "key_concepts": []
                })
    return normalized


def select_next_question_tot(
    role: str,
    performance_history: List[Any],
    question_bank: Any,
    covered_topics: Set[Any]
) -> Dict[str, Any]:
    """Use Tree-of-Thought approach to select the best next question.
    
    VIVA EXPLANATION:
    Tree-of-Thought (ToT) formulates question generation as a search tree over
    candidate instructional hypotheses. Instead of blindly picking the next question,
    ToT creates three competing strategic branches:
    - Branch A (Probe Weakness): Finds questions targeting past struggling topics (<70 score).
    - Branch B (Deepen Strength): Finds challenging questions extending past mastery (>=75 score).
    - Branch C (New Topic): Finds fresh topics to guarantee comprehensive syllabus coverage.
    
    The Interview Evaluation Policy inspects candidate momentum:
    - If last score < 60: Explores Branch A to remediate misconceptions.
    - If last score >= 80: Explores Branch B to test maximum proficiency.
    - Otherwise (or start): Explores Branch C to assess breadth.
    
    Args:
        role: Job role (e.g., 'python_developer')
        performance_history: List of (question, score) tuples or dicts
        question_bank: Dict or List of available questions
        covered_topics: Set of already-asked question IDs or topics
    
    Returns:
        dict with:
        - 'selected_question': The chosen question dict
        - 'branch': Which branch was selected ('weakness'/'strength'/'new_topic')
        - 'reasoning': Why this branch was chosen
        - 'all_branches': All 3 candidates for display
        - 'formatted_tot': Populated TOT_QUESTION_GENERATION_TEMPLATE string
    """
    normalized_bank = _normalize_question_bank(question_bank)
    
    # Identify asked questions and covered topics
    asked_ids: Set[Any] = set()
    asked_topic_names: Set[str] = set()
    
    for item in covered_topics:
        if isinstance(item, str):
            asked_topic_names.add(item.lower())
        asked_ids.add(item)
        
    for perf in performance_history:
        if isinstance(perf, (tuple, list)) and len(perf) >= 1:
            q_val = perf[0]
            if isinstance(q_val, dict):
                if "id" in q_val:
                    asked_ids.add(q_val["id"])
                if "topic" in q_val:
                    asked_topic_names.add(str(q_val["topic"]).lower())
            elif isinstance(q_val, (int, str)):
                asked_ids.add(q_val)
        elif isinstance(perf, dict):
            if "id" in perf:
                asked_ids.add(perf["id"])
            if "topic" in perf:
                asked_topic_names.add(str(perf["topic"]).lower())
                
    # Filter available unasked questions
    available_questions = [
        q for q in normalized_bank 
        if q.get("id") not in asked_ids and q.get("question") not in asked_ids
    ]
    
    # If exhausted, allow repeating from bank as fallback
    if not available_questions and normalized_bank:
        available_questions = list(normalized_bank)

    # Analyze performance history
    parsed_history: List[Dict[str, Any]] = []
    for entry in performance_history:
        if isinstance(entry, (tuple, list)) and len(entry) >= 2:
            q_info, score_val = entry[0], entry[1]
            topic_val = "general"
            if isinstance(q_info, dict):
                topic_val = q_info.get("topic", "general")
            elif isinstance(q_info, str):
                # Try finding topic from question bank
                for qb in normalized_bank:
                    if qb.get("question") == q_info or qb.get("id") == q_info:
                        topic_val = qb.get("topic", "general")
                        break
            parsed_history.append({"question": q_info, "score": int(score_val), "topic": topic_val})
        elif isinstance(entry, dict):
            parsed_history.append({
                "question": entry.get("question", "Unknown"),
                "score": int(entry.get("score", 70)),
                "topic": entry.get("topic", "general")
            })

    # Weakness topics (score < 70) and strength topics (score >= 75)
    weakness_topics = [p["topic"].lower() for p in parsed_history if p["score"] < 70]
    strength_topics = [p["topic"].lower() for p in parsed_history if p["score"] >= 75]

    all_topics_in_bank = {str(q.get("topic", "general")).lower() for q in normalized_bank}
    covered_topic_set = asked_topic_names.union({p["topic"].lower() for p in parsed_history})
    remaining_topics = all_topics_in_bank - covered_topic_set

    # ----------------------------------------------------
    # Generate Branch A: Probe Weakness
    # ----------------------------------------------------
    branch_a_q: Optional[Dict[str, Any]] = None
    branch_a_reason = ""
    
    if weakness_topics:
        target_topic = weakness_topics[-1]
        candidates = [q for q in available_questions if str(q.get("topic", "")).lower() == target_topic]
        # Prefer easy/medium to verify fundamentals
        easy_medium = [q for q in candidates if q.get("difficulty", "medium").lower() in {"easy", "medium"}]
        branch_a_q = easy_medium[0] if easy_medium else (candidates[0] if candidates else None)
        branch_a_reason = f"Probing identified weakness in topic '{target_topic}' to test foundational recovery."
        
    if not branch_a_q:
        # Fallback for Branch A: pick an easy foundational question
        easy_qs = [q for q in available_questions if q.get("difficulty", "").lower() == "easy"]
        branch_a_q = easy_qs[0] if easy_qs else (available_questions[0] if available_questions else None)
        branch_a_reason = "No severe weakness detected; testing core foundational knowledge."

    # ----------------------------------------------------
    # Generate Branch B: Deepen Strength
    # ----------------------------------------------------
    branch_b_q: Optional[Dict[str, Any]] = None
    branch_b_reason = ""
    
    if strength_topics:
        target_topic = strength_topics[-1]
        candidates = [q for q in available_questions if str(q.get("topic", "")).lower() == target_topic]
        # Prefer medium/hard to test depth
        hard_medium = [q for q in candidates if q.get("difficulty", "").lower() in {"hard", "medium"}]
        branch_b_q = hard_medium[0] if hard_medium else (candidates[0] if candidates else None)
        branch_b_reason = f"Deepening proven strength in '{target_topic}' with an advanced question."
        
    if not branch_b_q:
        # Fallback for Branch B: pick highest available difficulty question
        hard_qs = [q for q in available_questions if q.get("difficulty", "").lower() == "hard"]
        medium_qs = [q for q in available_questions if q.get("difficulty", "").lower() == "medium"]
        branch_b_q = hard_qs[0] if hard_qs else (medium_qs[0] if medium_qs else (available_questions[0] if available_questions else None))
        branch_b_reason = "Challenging candidate with higher complexity problem."

    # ----------------------------------------------------
    # Generate Branch C: New Topic
    # ----------------------------------------------------
    branch_c_q: Optional[Dict[str, Any]] = None
    branch_c_reason = ""
    
    if remaining_topics:
        candidates = [q for q in available_questions if str(q.get("topic", "")).lower() in remaining_topics]
        branch_c_q = candidates[0] if candidates else None
        target_topic = branch_c_q.get("topic", "new topic") if branch_c_q else "new topic"
        branch_c_reason = f"Broadening scope to untested syllabus topic '{target_topic}'."
        
    if not branch_c_q:
        # Pick any question not on the most recent topic
        recent_topic = parsed_history[-1]["topic"].lower() if parsed_history else ""
        diff_topic_qs = [q for q in available_questions if str(q.get("topic", "")).lower() != recent_topic]
        branch_c_q = diff_topic_qs[0] if diff_topic_qs else (available_questions[0] if available_questions else None)
        branch_c_reason = "Exploring new conceptual angle across syllabus."

    # ----------------------------------------------------
    # Tree-of-Thought Evaluation & Selection Policy
    # ----------------------------------------------------
    # Decide which branch is strategically best for the interview state
    if not parsed_history:
        selected_branch = "new_topic"
        selection_reasoning = (
            "Interview initialization: Selecting Branch C (New Topic) to evaluate "
            "baseline competency across a standard curriculum topic."
        )
    else:
        last_score = parsed_history[-1]["score"]
        avg_score = sum(p["score"] for p in parsed_history) / len(parsed_history)
        
        if last_score < 60 and branch_a_q is not None:
            selected_branch = "weakness"
            selection_reasoning = (
                f"Candidate struggled on the previous question (score: {last_score}/100). "
                f"Tree-of-Thought search evaluates Branch A (Probe Weakness) as highest utility "
                f"to diagnose underlying misconceptions before proceeding."
            )
        elif (last_score >= 80 or avg_score >= 80) and branch_b_q is not None:
            selected_branch = "strength"
            selection_reasoning = (
                f"Candidate exhibited strong mastery (recent score: {last_score}/100, avg: {avg_score:.1f}). "
                f"Tree-of-Thought search evaluates Branch B (Deepen Strength) as highest utility "
                f"to test architectural boundaries and upper-level capability."
            )
        else:
            selected_branch = "new_topic"
            selection_reasoning = (
                f"Candidate demonstrated steady performance (score: {last_score}/100). "
                f"Tree-of-Thought search evaluates Branch C (New Topic) as highest utility "
                f"to ensure balanced coverage across remaining required competencies."
            )

    branch_map = {
        "weakness": branch_a_q,
        "strength": branch_b_q,
        "new_topic": branch_c_q
    }

    # Safe fallback if chosen question is somehow None
    chosen_q = branch_map[selected_branch]
    if chosen_q is None:
        for fallback_branch, q_cand in branch_map.items():
            if q_cand is not None:
                selected_branch = fallback_branch
                chosen_q = q_cand
                selection_reasoning += f" (Fell back to Branch {fallback_branch} as primary branch had no questions)."
                break

    all_branches_info = {
        "branch_a": {
            "type": "weakness",
            "question": branch_a_q.get("question", "N/A") if branch_a_q else "No question available",
            "topic": branch_a_q.get("topic", "N/A") if branch_a_q else "N/A",
            "difficulty": branch_a_q.get("difficulty", "N/A") if branch_a_q else "N/A",
            "rationale": branch_a_reason
        },
        "branch_b": {
            "type": "strength",
            "question": branch_b_q.get("question", "N/A") if branch_b_q else "No question available",
            "topic": branch_b_q.get("topic", "N/A") if branch_b_q else "N/A",
            "difficulty": branch_b_q.get("difficulty", "N/A") if branch_b_q else "N/A",
            "rationale": branch_b_reason
        },
        "branch_c": {
            "type": "new_topic",
            "question": branch_c_q.get("question", "N/A") if branch_c_q else "No question available",
            "topic": branch_c_q.get("topic", "N/A") if branch_c_q else "N/A",
            "difficulty": branch_c_q.get("difficulty", "N/A") if branch_c_q else "N/A",
            "rationale": branch_c_reason
        }
    }

    # Prepare formatted ToT template
    perf_summary = (
        f"{len(parsed_history)} question(s) answered. "
        f"Average Score: {sum(p['score'] for p in parsed_history)/max(len(parsed_history), 1):.1f}/100"
        if parsed_history else "No questions completed yet"
    )

    formatted_tot = TOT_QUESTION_GENERATION_TEMPLATE.format(
        role=role.replace("_", " ").title(),
        performance_summary=perf_summary,
        topics_covered=", ".join(sorted(covered_topic_set)) if covered_topic_set else "None",
        topics_remaining=", ".join(sorted(remaining_topics)) if remaining_topics else "All topics touched",
        branch_a=f"[{all_branches_info['branch_a']['difficulty'].upper()}] {all_branches_info['branch_a']['question']}",
        branch_b=f"[{all_branches_info['branch_b']['difficulty'].upper()}] {all_branches_info['branch_b']['question']}",
        branch_c=f"[{all_branches_info['branch_c']['difficulty'].upper()}] {all_branches_info['branch_c']['question']}",
        selected=f"Branch '{selected_branch}': {chosen_q.get('question', 'None') if chosen_q else 'None'}",
        reasoning=selection_reasoning
    ).strip()

    return {
        "selected_question": chosen_q,
        "branch": selected_branch,
        "reasoning": selection_reasoning,
        "all_branches": all_branches_info,
        "formatted_tot": formatted_tot
    }


# ============================================================
# DEMONSTRATION & SELF-TEST (__main__)
# ============================================================
if __name__ == "__main__":
    print("=" * 75)
    print("SmartHire: Prompt Engineering Techniques (Unit 5 Generative AI)")
    print("=" * 75)

    # --------------------------------------------------------
    # DEMO 1: Interview Pattern Approach
    # --------------------------------------------------------
    print("\n" + "#" * 60)
    print("DEMO 1: INTERVIEW PATTERN APPROACH")
    print("#" * 60)
    
    sample_history = [
        ("Interviewer", "Welcome! Let's start with Python fundamentals. What is a list comprehension?"),
        ("Candidate", "A list comprehension is a concise syntax in Python for creating new lists based on existing iterables, like [x for x in range(10)].")
    ]
    
    interview_prompt = build_interview_prompt(
        role="python_developer",
        difficulty="medium",
        conversation_history=sample_history
    )
    print("\n[Generated Interview Prompt with Persona & Context Guardrails]:")
    print("-" * 50)
    print(interview_prompt)
    print("-" * 50)

    # --------------------------------------------------------
    # DEMO 2: Chain-of-Thought (CoT) Approach
    # --------------------------------------------------------
    print("\n" + "#" * 60)
    print("DEMO 2: CHAIN-OF-THOUGHT (CoT) EVALUATION")
    print("#" * 60)

    sample_question = "Explain how Python generators work and why they are memory efficient compared to regular functions."
    sample_key_concepts = ["yield", "lazy evaluation", "memory efficiency", "iterator", "state"]

    # Candidate 1: High Quality Response
    good_answer = (
        "In Python, a generator is created using a function containing the 'yield' statement instead of return. "
        "Unlike regular functions that calculate all values and store them in memory as a list, generators employ "
        "'lazy evaluation', producing items one at a time on demand. Because they yield values incrementally and maintain "
        "their execution state, they achieve tremendous memory efficiency, which is critical when processing large data streams."
    )

    # Candidate 2: Weak Response
    poor_answer = "Generators use yield. They are faster than normal functions."

    print("\n--- Evaluating Candidate A (Comprehensive Answer) ---")
    cot_result_a = build_cot_evaluation(sample_question, good_answer, sample_key_concepts)
    for step_desc in cot_result_a["steps"]:
        print(f"  > {step_desc}")
    print(f"  Final Score : {cot_result_a['score']}/100")
    print(f"  Rating      : {cot_result_a['rating']}")
    print(f"  Feedback    : {cot_result_a['feedback']}")

    print("\n--- Evaluating Candidate B (Shallow Answer) ---")
    cot_result_b = build_cot_evaluation(sample_question, poor_answer, sample_key_concepts)
    for step_desc in cot_result_b["steps"]:
        print(f"  > {step_desc}")
    print(f"  Final Score : {cot_result_b['score']}/100")
    print(f"  Rating      : {cot_result_b['rating']}")
    print(f"  Feedback    : {cot_result_b['feedback']}")

    print("\n--- Fully Rendered CoT Evaluation Template Output ---")
    print(cot_result_a["formatted_cot"])

    # --------------------------------------------------------
    # DEMO 3: Tree-of-Thought (ToT) Approach
    # --------------------------------------------------------
    print("\n" + "#" * 60)
    print("DEMO 3: TREE-OF-THOUGHT (ToT) QUESTION SELECTION")
    print("#" * 60)

    # Curated Question Bank
    demo_question_bank = {
        0: {
            "id": 0,
            "question": "What is the difference between mutable and immutable types in Python?",
            "topic": "data_structures",
            "difficulty": "easy",
            "key_concepts": ["id", "hashable", "list vs tuple", "mutation"]
        },
        1: {
            "id": 1,
            "question": "Explain method resolution order (MRO) and the C3 linearization algorithm in Python multiple inheritance.",
            "topic": "oop",
            "difficulty": "hard",
            "key_concepts": ["mro", "c3 linearization", "super()", "diamond problem"]
        },
        2: {
            "id": 2,
            "question": "How does the Python Global Interpreter Lock (GIL) impact multithreaded CPU-bound programs?",
            "topic": "concurrency",
            "difficulty": "medium",
            "key_concepts": ["gil", "cpu-bound", "io-bound", "multiprocessing"]
        },
        3: {
            "id": 3,
            "question": "Explain async/await syntax and how the asyncio event loop schedules coroutines.",
            "topic": "concurrency",
            "difficulty": "hard",
            "key_concepts": ["event loop", "coroutine", "future", "task", "non-blocking"]
        },
        4: {
            "id": 4,
            "question": "How do Python decorators work under the hood using first-class functions and closures?",
            "topic": "functional_programming",
            "difficulty": "medium",
            "key_concepts": ["closure", "wrapper", "first-class function", "@functools.wraps"]
        }
    }

    # Scenario: Candidate excelled in OOP (score: 92) but struggled in Concurrency (score: 48)
    simulated_history = [
        ({"id": 1, "topic": "oop", "question": "Explain MRO in multiple inheritance."}, 92),
        ({"id": 2, "topic": "concurrency", "question": "Explain the Python GIL."}, 48)
    ]
    covered_set = {1, 2}

    tot_selection = select_next_question_tot(
        role="python_developer",
        performance_history=simulated_history,
        question_bank=demo_question_bank,
        covered_topics=covered_set
    )

    print("\n[Tree-of-Thought Decision Tree]:")
    print(f"Branch A (Weakness Probe)  : {tot_selection['all_branches']['branch_a']['question']}")
    print(f"Branch B (Strength Deepen) : {tot_selection['all_branches']['branch_b']['question']}")
    print(f"Branch C (New Topic)       : {tot_selection['all_branches']['branch_c']['question']}")
    print("-" * 50)
    print(f"--> Selected Branch : {tot_selection['branch'].upper()}")
    print(f"--> Selected Question: {tot_selection['selected_question']['question']}")
    print(f"--> Strategic Reason : {tot_selection['reasoning']}")
    print("-" * 50)

    print("\n[Rendered Tree-of-Thought Prompt/Summary Template]:")
    print(tot_selection["formatted_tot"])
    print("\n" + "=" * 75)
    print("All Unit 5 Prompt Engineering demonstrations completed successfully!")
    print("=" * 75)
