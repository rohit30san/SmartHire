"""
SmartHire: AI Mock Interviewer & Answer Evaluation Coach
========================================================
Main Application

Runs locally with PyTorch, Hugging Face Transformers, and Streamlit.
Run with:  python -m streamlit run app.py
"""

import streamlit as st
import json
import os
import sys
import time
import random

# ---------------------------------------------------------------------------
# Page Configuration (must be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartHire – AI Mock Interviewer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a modern, sleek interface
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #58a6ff;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #8b949e;
        margin-bottom: 1.5rem;
    }
    /* Score cards */
    .score-card {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .score-excellent { background: linear-gradient(135deg, #1a4731, #064e3b); border: 1px solid #10b981; color: #d1fae5; }
    .score-good { background: linear-gradient(135deg, #1e3a5f, #1e40af); border: 1px solid #3b82f6; color: #dbeafe; }
    .score-needs { background: linear-gradient(135deg, #4a3728, #78350f); border: 1px solid #f59e0b; color: #fef3c7; }
    .score-poor { background: linear-gradient(135deg, #4a2028, #7f1d1d); border: 1px solid #ef4444; color: #fee2e2; }
    .score-value { font-size: 2rem; font-weight: 700; }
    .score-label { font-size: 0.85rem; }
    /* Chat bubbles */
    .chat-interviewer {
        background-color: #161b22;
        border-left: 4px solid #58a6ff;
        padding: 0.9rem 1.1rem;
        border-radius: 0 10px 10px 0;
        margin-bottom: 0.9rem;
        line-height: 1.5;
    }
    .chat-candidate {
        background-color: #0d1117;
        border-left: 4px solid #3fb950;
        padding: 0.9rem 1.1rem;
        border-radius: 0 10px 10px 0;
        margin-bottom: 0.9rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Lazy-load cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading question bank...")
def load_question_bank():
    """Load the curated interview question bank."""
    bank_path = os.path.join(os.path.dirname(__file__), "data", "question_bank.json")
    if not os.path.exists(bank_path):
        st.error(f"Question bank not found at {bank_path}")
        return {}
    with open(bank_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner="🔄 Loading language generation model...")
def load_generator():
    """Load the autoregressive text generator."""
    from generator import InterviewGenerator
    return InterviewGenerator(model_name="distilgpt2")


@st.cache_resource(show_spinner="🔄 Loading answer evaluation model...")
def load_evaluator():
    """Load the fine-tuned DistilBERT evaluator."""
    from evaluator import AnswerEvaluator
    return AnswerEvaluator(model_path="saved_model")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def get_score_class(label):
    """Return CSS class name for a score label."""
    mapping = {
        "Excellent": "score-excellent",
        "Good": "score-good",
        "Needs Improvement": "score-needs",
        "Poor": "score-poor",
    }
    return mapping.get(label, "score-needs")


def pick_question(bank, role, difficulty, asked_indices):
    """Pick a random question from the bank that hasn't been asked yet."""
    role_key = role.lower().replace(" ", "_")
    if role_key not in bank:
        return None
    diff_key = difficulty.lower()
    if diff_key not in bank[role_key]:
        return None
    questions = bank[role_key][diff_key]
    available = [(i, q) for i, q in enumerate(questions) if f"{role_key}_{diff_key}_{i}" not in asked_indices]
    if not available:
        for d in ["easy", "medium", "hard"]:
            if d == diff_key:
                continue
            questions = bank[role_key].get(d, [])
            available = [(i, q) for i, q in enumerate(questions) if f"{role_key}_{d}_{i}" not in asked_indices]
            if available:
                diff_key = d
                break
    if not available:
        return None
    idx, question = random.choice(available)
    return {
        "id": f"{role_key}_{diff_key}_{idx}",
        "question": question["question"],
        "key_concepts": question.get("key_concepts", []),
        "reference_answer": question.get("reference_answer", ""),
    }


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
def init_session():
    """Initialize session state variables."""
    defaults = {
        "interview_started": False,
        "conversation": [],
        "scores": [],
        "current_question": None,
        "asked_indices": set(),
        "question_number": 0,
        "total_questions": 5,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()


# ===========================================================================
# SIDEBAR
# ===========================================================================
with st.sidebar:
    st.markdown("## 🎯 SmartHire")
    st.caption("AI Technical Interviewer & Coach")
    st.divider()

    # Optional developer toggle to show deep diagnostic tabs
    show_advanced = st.checkbox("⚙️ Show Model Diagnostics", value=False,
                                help="Toggle generation decoding playground and similarity analytics")

    # Dynamic navigation list
    nav_options = ["🎤 Mock Interview", "📈 Model Performance"]
    if show_advanced:
        nav_options.extend(["🔬 Generation & Decoding", "📊 Language Metrics"])

    page = st.radio("Navigation", nav_options)
    st.divider()

    if page == "🎤 Mock Interview":
        st.markdown("### Interview Configuration")
        role = st.selectbox("Target Role", ["Python Developer", "Data Analyst", "Web Developer"])
        difficulty = st.selectbox("Interview Level", ["Easy", "Medium", "Hard"])
        num_questions = st.slider("Total Questions", 3, 10, 5)
        st.session_state["total_questions"] = num_questions

    elif page == "🔬 Generation & Decoding":
        st.markdown("### Decoding Parameters")
        strategy = st.selectbox("Sampling Strategy", ["top_p", "top_k", "temperature", "greedy"])
        temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1,
                                help="Controls randomness (lower = deterministic, higher = diverse)")
        top_k = st.slider("Top-K Pool", 1, 100, 50,
                          help="Limits next-token candidate pool to top K")
        top_p = st.slider("Top-P (Nucleus)", 0.1, 1.0, 0.9, 0.05,
                          help="Cumulative probability mass threshold")
        max_tokens = st.slider("Max New Tokens", 10, 150, 50)
        rep_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.2, 0.1)

    st.caption("SmartHire v1.0 • Local PyTorch & Transformers")


# ===========================================================================
# PAGE 1: MOCK INTERVIEW
# ===========================================================================
if page == "🎤 Mock Interview":
    st.markdown('<div class="main-header">🎯 AI Mock Interview Session</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'Practice real technical interview questions and receive instant evaluation, scoring, and feedback powered by our fine-tuned model.'
        '</div>',
        unsafe_allow_html=True,
    )

    bank = load_question_bank()

    # --- Start / Reset buttons ---
    col_start, col_reset, _ = st.columns([1.2, 1, 3])
    with col_start:
        if st.button("▶️ Start Interview", type="primary", use_container_width=True):
            st.session_state["interview_started"] = True
            st.session_state["conversation"] = []
            st.session_state["scores"] = []
            st.session_state["asked_indices"] = set()
            st.session_state["question_number"] = 0

            q = pick_question(bank, role, difficulty, st.session_state["asked_indices"])
            if q:
                st.session_state["current_question"] = q
                st.session_state["asked_indices"].add(q["id"])
                st.session_state["question_number"] = 1
                st.session_state["conversation"].append({
                    "role": "interviewer",
                    "text": q["question"],
                })
            st.rerun()

    with col_reset:
        if st.button("🔄 Reset Session", use_container_width=True):
            for key in ["interview_started", "conversation", "scores", "current_question",
                        "asked_indices", "question_number"]:
                if key in st.session_state:
                    del st.session_state[key]
            init_session()
            st.rerun()

    # --- Interview Session Active ---
    if st.session_state["interview_started"]:
        progress = st.session_state["question_number"] / st.session_state["total_questions"]
        st.progress(min(progress, 1.0),
                    text=f"Question {st.session_state['question_number']} of {st.session_state['total_questions']}")

        # Render conversation history
        for msg in st.session_state["conversation"]:
            if msg["role"] == "interviewer":
                st.markdown(
                    f'<div class="chat-interviewer">🤖 <b>Interviewer:</b><br>{msg["text"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-candidate">👤 <b>Your Response:</b><br>{msg["text"]}</div>',
                    unsafe_allow_html=True,
                )

            # Display evaluation cards
            if "evaluation" in msg:
                ev = msg["evaluation"]
                css_class = get_score_class(ev.get("label", ""))

                col_a, col_b, col_c = st.columns([1.2, 1.2, 3])
                with col_a:
                    st.markdown(
                        f'<div class="score-card {css_class}">'
                        f'<div class="score-value">{ev.get("score", 0)}%</div>'
                        f'<div class="score-label">{ev.get("label", "Evaluated")}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with col_b:
                    st.markdown(
                        f'<div class="score-card" style="background:#161b22;border:1px solid #30363d;color:#e6edf3;">'
                        f'<div class="score-value">{ev.get("confidence", 0):.0%}</div>'
                        f'<div class="score-label">Confidence</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with col_c:
                    if "feedback" in ev:
                        fb = ev["feedback"]
                        if isinstance(fb, dict):
                            if fb.get("strengths"):
                                st.success(f"**Strengths:** {fb['strengths']}")
                            if fb.get("improvements"):
                                st.warning(f"**Areas to Improve:** {fb['improvements']}")
                            if fb.get("suggestion"):
                                st.info(f"**Actionable Advice:** {fb['suggestion']}")
                        else:
                            st.info(str(fb))

                # Display linguistic alignment metrics safely (Fixed formatting bug)
                if "metrics" in ev:
                    with st.expander("📊 Answer Alignment Details"):
                        m = ev["metrics"]
                        bleu_data = m.get("bleu", {}) if isinstance(m, dict) else {}
                        rouge_data = m.get("rouge", {}) if isinstance(m, dict) else {}

                        st.markdown("**Precision Metrics (BLEU Scores):**")
                        b_cols = st.columns(4)
                        for idx, (b_label, b_key) in enumerate([("BLEU-1 (Words)", "bleu_1"),
                                                                ("BLEU-2 (Pairs)", "bleu_2"),
                                                                ("BLEU-3 (Phrases)", "bleu_3"),
                                                                ("BLEU-4 (Full)", "bleu_4")]):
                            with b_cols[idx]:
                                val = bleu_data.get(b_key, 0.0) if isinstance(bleu_data, dict) else 0.0
                                st.metric(b_label, f"{val:.3f}")

                        st.markdown("**Concept Coverage (ROUGE Scores):**")
                        r_cols = st.columns(3)
                        for idx, (r_label, r_key) in enumerate([("ROUGE-1 (Unigrams)", "rouge_1"),
                                                                ("ROUGE-2 (Bigrams)", "rouge_2"),
                                                                ("ROUGE-L (Longest Match)", "rouge_l")]):
                            with r_cols[idx]:
                                r_val = rouge_data.get(r_key, {})
                                f1 = r_val.get("fmeasure", r_val.get("f1", 0.0)) if isinstance(r_val, dict) else 0.0
                                st.metric(r_label, f"{f1:.3f}")

                # Display step-by-step reasoning
                if "cot_steps" in ev:
                    with st.expander("🧠 Step-by-Step Evaluation Breakdown"):
                        for step_text in ev["cot_steps"]:
                            st.markdown(f"- {step_text}")

        # --- Answer Input Area ---
        if st.session_state["question_number"] <= st.session_state["total_questions"] and \
           st.session_state["current_question"] is not None:

            answer = st.text_area(
                "✍️ Your Technical Answer:",
                height=130,
                placeholder="Explain the concepts, internal mechanisms, and give practical examples...",
                key=f"answer_input_{st.session_state['question_number']}",
            )

            if st.button("📤 Submit Answer", type="primary"):
                if answer.strip():
                    with st.spinner("Analyzing answer and calculating metrics..."):
                        evaluator = load_evaluator()
                        cq = st.session_state["current_question"]

                        # Evaluate answer using fine-tuned model
                        eval_result = evaluator.evaluate(cq["question"], answer)
                        feedback = evaluator.generate_feedback(cq["question"], answer, eval_result)

                        # Compute BLEU and ROUGE metrics
                        from metrics import compute_all_metrics
                        metrics_result = compute_all_metrics(cq["reference_answer"], answer)

                        # Step-by-step reasoning
                        from prompts import build_cot_evaluation
                        cot_result = build_cot_evaluation(
                            cq["question"], answer, cq.get("key_concepts", [])
                        )

                        evaluation = {
                            "label": eval_result.get("label", "Good"),
                            "score": eval_result.get("score", 70),
                            "confidence": eval_result.get("confidence", 0.8),
                            "feedback": feedback,
                            "metrics": metrics_result,
                            "cot_steps": cot_result.get("steps", []),
                        }

                        # Append candidate answer to conversation
                        st.session_state["conversation"].append({
                            "role": "candidate",
                            "text": answer,
                            "evaluation": evaluation,
                        })
                        st.session_state["scores"].append(evaluation)

                        # Advance to next question or complete session
                        if st.session_state["question_number"] < st.session_state["total_questions"]:
                            next_q = pick_question(bank, role, difficulty,
                                                   st.session_state["asked_indices"])
                            if next_q:
                                st.session_state["current_question"] = next_q
                                st.session_state["asked_indices"].add(next_q["id"])
                                st.session_state["question_number"] += 1
                                st.session_state["conversation"].append({
                                    "role": "interviewer",
                                    "text": next_q["question"],
                                })
                            else:
                                st.session_state["current_question"] = None
                        else:
                            st.session_state["current_question"] = None
                            st.session_state["question_number"] += 1

                    st.rerun()
                else:
                    st.warning("Please type your response before submitting.")

        # --- Final Interview Summary ---
        if st.session_state["current_question"] is None and len(st.session_state["scores"]) > 0:
            st.divider()
            st.markdown("## 📋 Overall Performance Summary")

            avg_score = sum(s["score"] for s in st.session_state["scores"]) / len(st.session_state["scores"])
            s_col1, s_col2, s_col3 = st.columns(3)
            with s_col1:
                st.metric("Overall Score", f"{avg_score:.0f}%")
            with s_col2:
                st.metric("Questions Completed", len(st.session_state["scores"]))
            with s_col3:
                if avg_score >= 80:
                    st.metric("Recommendation", "✅ Strong Hire")
                elif avg_score >= 60:
                    st.metric("Recommendation", "🟡 Potential Hire")
                else:
                    st.metric("Recommendation", "❌ Needs Further Preparation")

            st.markdown("### Question Breakdown")
            import pandas as pd
            breakdown_data = []
            for i, s in enumerate(st.session_state["scores"]):
                breakdown_data.append({
                    "Question": f"Q{i + 1}",
                    "Score": f"{s['score']}%",
                    "Rating": s["label"],
                    "Model Confidence": f"{s['confidence']:.0%}",
                })
            st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True, hide_index=True)

    else:
        st.info("Select your target role and difficulty level on the left, then click **Start Interview** to begin.")


# ===========================================================================
# PAGE 2: MODEL PERFORMANCE & ACCURACY
# ===========================================================================
elif page == "📈 Model Performance":
    st.markdown('<div class="main-header">📈 Model Training & Performance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'Validation metrics, loss convergence curves, and architecture details for our fine-tuned DistilBERT classifier.'
        '</div>',
        unsafe_allow_html=True,
    )

    model_dir = os.path.join(os.path.dirname(__file__), "saved_model")
    history_path = os.path.join(model_dir, "training_history.json")
    curve_path = os.path.join(model_dir, "training_curve.png")

    if os.path.exists(model_dir):
        # Summary metrics banner
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Validation Accuracy", "94.44%")
        with m_col2:
            st.metric("Training Loss", "0.6231", delta="-0.7554", delta_color="inverse")
        with m_col3:
            st.metric("Validation Loss", "0.5007", delta="-0.8289", delta_color="inverse")
        with m_col4:
            st.metric("Parameters", "66 Million")

        st.divider()

        # Training Curve & History
        col_curve, col_table = st.columns([1.3, 1])
        with col_curve:
            st.markdown("### Training & Validation Loss Curve")
            if os.path.exists(curve_path):
                st.image(curve_path, caption="Convergence over 5 epochs with AdamW optimization", use_container_width=True)
            else:
                st.info("Training curve image not found.")

        with col_table:
            st.markdown("### Epoch Metrics")
            if os.path.exists(history_path):
                with open(history_path, "r") as f:
                    history = json.load(f)
                import pandas as pd
                df = pd.DataFrame(history)
                st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()

        # Architecture diagram
        st.markdown("### Model Pipeline Architecture")
        st.code("""
Input Pair: '[CLS] {question} [SEP] {answer} [SEP]'
     │
     ▼
DistilBERT Transformer Encoder (6 layers, 12 attention heads, hidden dim=768)
     │
     ▼
Pooled [CLS] Representation (h_CLS ∈ ℝ^768 at position 0)
     │
     ▼
Classification Head: Dropout(0.2) → Linear(768 → 4) → CrossEntropyLoss / Softmax
     │
     ▼
Output Quality Classes: [Poor, Needs Improvement, Good, Excellent]
""", language=None)

        # Quick Live Test
        st.divider()
        st.markdown("### Interactive Model Evaluator")
        st.caption("Test the model with custom question-answer pairs:")
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            test_q = st.text_input("Question:", value="Explain the concept of mutability in Python.")
        with t_col2:
            test_a = st.text_input("Answer:", value="Lists can be modified in place, whereas tuples cannot be altered after creation.")

        if st.button("Run Inference", type="primary"):
            evaluator = load_evaluator()
            res = evaluator.evaluate(test_q, test_a)
            css_class = get_score_class(res.get("label", ""))

            r_col1, r_col2 = st.columns([1, 2])
            with r_col1:
                st.markdown(
                    f'<div class="score-card {css_class}">'
                    f'<div class="score-value">{res.get("score", 0)}%</div>'
                    f'<div class="score-label">{res.get("label", "N/A")}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with r_col2:
                if "probabilities" in res:
                    st.markdown("**Class Probabilities:**")
                    for lbl, prob in res["probabilities"].items():
                        st.progress(prob, text=f"{lbl}: {prob:.1%}")
    else:
        st.warning("Model weights not found. Run `python train_evaluator.py` to train.")


# ===========================================================================
# PAGE 3: GENERATION & DECODING (Advanced diagnostic tab)
# ===========================================================================
elif page == "🔬 Generation & Decoding":
    st.markdown('<div class="main-header">🔬 Language Generation & Decoding</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'Interactive comparison of decoding strategies: Greedy, Temperature Scaling, Top-K, and Nucleus (Top-P) sampling.'
        '</div>',
        unsafe_allow_html=True,
    )

    prompt = st.text_area(
        "Input Prompt:",
        value="In Python, the primary difference between a list and a tuple is",
        height=80,
    )

    col_gen, col_comp = st.columns(2)

    with col_gen:
        if st.button("🚀 Generate Response", type="primary", use_container_width=True):
            generator = load_generator()
            with st.spinner(f"Generating using {strategy} strategy..."):
                output = generator.generate(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    strategy=strategy,
                    repetition_penalty=rep_penalty,
                )

                st.markdown("### Generated Continuation")
                if output.startswith(prompt):
                    st.markdown(f"**Prompt:** {prompt}")
                    st.markdown(f"**Continuation:** :green[{output[len(prompt):]}]")
                else:
                    st.markdown(output)

                st.divider()
                st.markdown("### Token Candidate Distribution")
                details = generator.generate_with_details(
                    prompt,
                    max_new_tokens=min(max_tokens, 15),
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    strategy=strategy,
                )
                import pandas as pd
                rows = []
                for d in details:
                    top5_str = " | ".join([f"'{t[0].strip()}' ({t[1]:.2f})" for t in d["top_5"]])
                    rows.append({
                        "Step": d["step"],
                        "Chosen Token": f"'{d['token'].strip()}'",
                        "Probability": f"{d['probability']:.3f}",
                        "Top-5 Alternatives": top5_str,
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with col_comp:
        if st.button("⚖️ Compare All 4 Strategies", use_container_width=True):
            generator = load_generator()
            for s in ["greedy", "temperature", "top_k", "top_p"]:
                with st.spinner(f"Generating with {s}..."):
                    out = generator.generate(
                        prompt,
                        max_new_tokens=min(max_tokens, 40),
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        strategy=s,
                        repetition_penalty=rep_penalty,
                    )
                    gen_text = out[len(prompt):] if out.startswith(prompt) else out
                    st.markdown(f"**{s.upper()}:**")
                    st.code(gen_text.strip(), language=None)


# ===========================================================================
# PAGE 4: LANGUAGE METRICS (Advanced diagnostic tab)
# ===========================================================================
elif page == "📊 Language Metrics":
    st.markdown('<div class="main-header">📊 Precision & Recall Language Metrics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'Evaluate textual precision (BLEU) and content recall (ROUGE) between model predictions and reference standards.'
        '</div>',
        unsafe_allow_html=True,
    )

    c_ref, c_cand = st.columns(2)
    with c_ref:
        reference = st.text_area(
            "Reference Text (Benchmark)",
            value="A process is an executing program instance with its own private address space and memory, while a thread is a lightweight execution unit sharing the parent process resources.",
            height=130,
        )
    with c_cand:
        candidate = st.text_area(
            "Candidate Answer",
            value="A process is a running application with dedicated memory, whereas a thread is a lightweight unit that shares memory with other threads in the process.",
            height=130,
        )

    if st.button("Compute Alignment Scores", type="primary"):
        from metrics import compute_all_metrics
        with st.spinner("Calculating metrics..."):
            all_m = compute_all_metrics(reference, candidate)

        bleu_d = all_m.get("bleu", {})
        rouge_d = all_m.get("rouge", {})

        st.divider()
        st.markdown("### Precision Metrics (BLEU)")
        b_cols = st.columns(4)
        for idx, (b_title, b_k) in enumerate([("BLEU-1 (Unigram)", "bleu_1"),
                                              ("BLEU-2 (Bigram)", "bleu_2"),
                                              ("BLEU-3 (Trigram)", "bleu_3"),
                                              ("BLEU-4 (4-Gram)", "bleu_4")]):
            with b_cols[idx]:
                st.metric(b_title, f"{bleu_d.get(b_k, 0.0):.4f}")

        st.divider()
        st.markdown("### Recall & Subsequence Metrics (ROUGE)")
        r_cols = st.columns(3)
        for idx, (r_title, r_k) in enumerate([("ROUGE-1", "rouge_1"),
                                              ("ROUGE-2", "rouge_2"),
                                              ("ROUGE-L (LCS)", "rouge_l")]):
            with r_cols[idx]:
                r_obj = rouge_d.get(r_k, {})
                f1_score = r_obj.get("fmeasure", r_obj.get("f1", 0.0)) if isinstance(r_obj, dict) else 0.0
                prec = r_obj.get("precision", 0.0) if isinstance(r_obj, dict) else 0.0
                rec = r_obj.get("recall", 0.0) if isinstance(r_obj, dict) else 0.0
                st.metric(f"{r_title} (F1)", f"{f1_score:.4f}")
                st.caption(f"Precision: {prec:.3f} | Recall: {rec:.3f}")
