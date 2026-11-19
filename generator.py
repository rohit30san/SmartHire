"""
Interview Generator Module
==========================
Custom Autoregressive Text Generation using DistilGPT-2.

This module implements the token generation loop and decoding algorithms
FROM SCRATCH using PyTorch and Hugging Face Transformers.

Academic Context / Viva Notes:
------------------------------
Standard inference in libraries like Hugging Face typically calls `model.generate()`,
which conceals the mathematical operations behind text decoding. This module
exposes the full inference pipeline for educational purposes:
1. Autoregressive Loop:
   Predicting one token at a time conditioned on all preceding tokens:
   P(w_1, w_2, ..., w_T) = prod_{t=1}^T P(w_t | w_1, ..., w_{t-1})
2. Logit Processing:
   - Repetition Penalty: Penalizes logits of previously generated tokens.
   - Temperature Scaling: Modulates the entropy of the probability distribution.
3. Truncation / Filtering:
   - Top-K Sampling: Truncates the vocabulary to the K most probable tokens.
   - Top-P (Nucleus) Sampling: Truncates to the smallest set of tokens whose
     cumulative probability mass exceeds threshold p.
4. Sampling:
   - Categorical sampling using torch.multinomial or deterministic greedy argmax.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class InterviewGenerator:
    """Custom autoregressive text generator using DistilGPT-2.

    Implements decoding strategies from scratch to demonstrate
    understanding of language model inference (Lab 3 & 4 of syllabus).
    """

    def __init__(self, model_name: str = 'distilgpt2', device: Optional[str] = None):
        """Initialize the tokenizer and causal language model.

        VIVA EXPLANATION:
        -----------------
        - DistilGPT-2: A distilled version of OpenAI's GPT-2 (English). It has
          6 transformer layers, 82 million parameters (vs. 124M in GPT-2 small),
          and runs approximately 2x faster while retaining ~95% of performance.
        - Causal LM: Uses masked multi-head self-attention where each position
          can only attend to itself and prior positions (autoregressive property).
        - Pad Token: GPT-2 was trained without an explicit padding token. Setting
          pad_token = eos_token allows batch encoding when needed without errors.
        """
        print(f"Loading model '{model_name}' and tokenizer...")
        
        # Determine target compute device (GPU if available, otherwise CPU)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Using compute device: {self.device}")

        # Load tokenizer and causal language model from Hugging Face Hub
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # GPT-2 does not define a pad_token by default; assign eos_token as pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Set to evaluation mode (disables dropout layers for deterministic evaluation)
        self.model.eval()
        print(f"Model '{model_name}' loaded successfully on {self.device}.\n")

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        penalty: float
    ) -> torch.Tensor:
        """Apply repetition penalty to discourage repeating already generated tokens.

        VIVA EXPLANATION:
        -----------------
        Autoregressive language models often fall into degenerative loops, repeating
        the same word or phrase (e.g., 'and then and then and then').
        
        The repetition penalty (Keskar et al., 2019 - CTRL paper) modifies the logits
        of all tokens that have already appeared in the input sequence:
        - If logit > 0: logit = logit / penalty  (diminishes positive logit)
        - If logit < 0: logit = logit * penalty  (pushes negative logit further negative)
        
        A penalty > 1.0 penalizes repeats. A penalty = 1.0 applies no penalty.
        """
        if penalty == 1.0 or penalty <= 0.0:
            return logits

        modified_logits = logits.clone()
        
        # Collect set of unique token IDs that have appeared in the current sequence
        unique_tokens = torch.unique(input_ids)
        
        for token_id in unique_tokens:
            # Check the sign of the logit for each token
            current_logit = modified_logits[:, token_id]
            modified_logits[:, token_id] = torch.where(
                current_logit > 0,
                current_logit / penalty,
                current_logit * penalty
            )

        return modified_logits

    def _apply_temperature(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Scale logits by temperature before softmax.

        VIVA EXPLANATION:
        -----------------
        Temperature scaling modifies the sharpness of the probability distribution:
            P(w_i) = exp(z_i / T) / sum_j exp(z_j / T)
        
        Mathematical intuition:
        - As T -> 0: The highest logit dominates completely. In the limit T -> 0,
          the distribution becomes a one-hot vector at the argmax (Greedy Search).
        - When T < 1.0 (e.g., 0.5): Discrepancies between high and low logits are amplified.
          The distribution is 'sharper' or 'peakier'. Model generates more confident,
          predictable, and coherent text.
        - When T = 1.0: Normal unscaled model probabilities.
        - When T > 1.0 (e.g., 1.5): Differences between logits are compressed.
          The distribution becomes 'flatter' (higher entropy), closer to uniform.
          Model produces more diverse, creative, but potentially chaotic text.
        """
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be strictly positive, got {temperature}")
        
        # Element-wise division by temperature scalar
        return logits / temperature

    def _top_k_filtering(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Keep only top-k highest probability tokens, set rest to -infinity.

        VIVA EXPLANATION:
        -----------------
        Top-K Sampling (Fan et al., 2018):
        In standard vocabulary sampling, the vocabulary has ~50,257 tokens. The 'long tail'
        consists of thousands of low-probability, nonsensical tokens. If even one is sampled,
        the generation can degrade.
        
        Top-K solves this by:
        1. Ranking all vocabulary tokens by their logit/probability.
        2. Retaining only the top K tokens (e.g., K = 50).
        3. Setting the logits of all remaining tokens to -infinity (-inf).
        4. Re-normalizing with softmax so probabilities of the top-K sum to 1.
        
        Limitation for Viva:
        K is fixed. If the model is 99% confident in 1 word, Top-K still samples from K words.
        If the model is uncertain across 200 words, Top-K prematurely cuts off after K words.
        """
        # Validate K
        if k <= 0 or k >= logits.size(-1):
            return logits

        filtered_logits = logits.clone()
        
        # Identify the cutoff value: the k-th highest logit value
        # torch.topk returns (values, indices) sorted descending
        topk_values, _ = torch.topk(filtered_logits, k, dim=-1)
        cutoff_value = topk_values[..., -1, None]  # Shape: (batch_size, 1)

        # Set all logits below the cutoff to -inf
        indices_to_remove = filtered_logits < cutoff_value
        filtered_logits[indices_to_remove] = -float('Inf')

        return filtered_logits

    def _top_p_filtering(self, logits: torch.Tensor, p: float, min_tokens_to_keep: int = 1) -> torch.Tensor:
        """Nucleus sampling: keep smallest set of tokens whose cumulative probability exceeds p.

        VIVA EXPLANATION:
        -----------------
        Top-P / Nucleus Sampling (Holtzman et al., 2019):
        Instead of fixing the candidate count K, Top-P dynamically adjusts the candidate pool
        based on model confidence by setting a cumulative probability threshold 'p' (e.g., 0.90).
        
        Algorithm:
        1. Sort logits in descending order.
        2. Convert to probabilities using Softmax: P = softmax(sorted_logits).
        3. Compute cumulative probabilities: C_i = sum_{j=1}^i P_j.
        4. Keep all tokens until the cumulative sum C_i exceeds p.
        5. Mask out all subsequent tokens by setting their logits to -inf.
        
        Why Nucleus Sampling is superior to Top-K:
        - Confident context: E.g., 'The capital of France is...' -> 'Paris' has 0.95 prob.
          The nucleus size collapses to 1 token.
        - Uncertain context: E.g., 'I think that...' -> Probabilities are distributed
          broadly over dozens of tokens. The nucleus expands to include 40+ tokens.
        """
        if p >= 1.0 or p <= 0.0:
            return logits

        filtered_logits = logits.clone()

        # Step 1: Sort logits descending
        sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True, dim=-1)

        # Step 2: Compute softmax probabilities on sorted logits
        sorted_probs = F.softmax(sorted_logits, dim=-1)

        # Step 3: Compute cumulative probability sum
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Step 4: Mask tokens where cumulative probability exceeds threshold p
        # Shift mask to the right by 1 so we keep the first token that crosses threshold p
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False  # Always keep at least the top-1 token

        # Enforce keeping at least `min_tokens_to_keep` tokens
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = False

        # Step 5: Scatter the removal mask back to original vocabulary positions
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        filtered_logits[indices_to_remove] = -float('Inf')

        return filtered_logits

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        strategy: str = 'top_p',
        repetition_penalty: float = 1.2
    ) -> str:
        """Generate text token-by-token using custom decoding.

        VIVA EXPLANATION:
        -----------------
        Autoregressive Generation Pipeline:
        1. Tokenize prompt text into token IDs.
        2. Feed tokens into the causal language model.
        3. Model outputs unnormalized log-odds (logits) of size [batch, seq_len, vocab_size].
        4. Extract the logits for the final position (t): logits for next token prediction.
        5. Apply repetition penalty for tokens already generated.
        6. Apply decoding strategy:
           - 'greedy': Pick argmax (most probable token deterministically).
           - 'temperature': Scale logits and sample directly from categorical distribution.
           - 'top_k': Filter down to top-k candidates, re-normalize, and sample.
           - 'top_p': Filter down to nucleus of cumulative probability p, re-normalize, sample.
        7. Append selected token to the input sequence.
        8. Repeat until max_new_tokens is reached or EOS (End Of Sequence) token is generated.

        Args:
            prompt: Input text prompt to condition the generation.
            max_new_tokens: Maximum number of new tokens to produce.
            temperature: Randomness factor (>0). Lower = focused, Higher = creative.
            top_k: Number of highest-probability tokens to retain for top_k strategy.
            top_p: Cumulative probability threshold for nucleus (top_p) strategy.
            strategy: One of 'greedy', 'temperature', 'top_k', 'top_p'.
            repetition_penalty: Penalty factor for repeating tokens (>1.0 reduces loops).

        Returns:
            The complete decoded text including prompt and newly generated tokens.
        """
        # Step 1: Tokenize prompt input
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        generated_ids = input_ids.clone()

        # Step 2: Autoregressive token generation loop
        for _ in range(max_new_tokens):
            with torch.no_grad():
                # Forward pass: forward the full sequence generated so far
                outputs = self.model(generated_ids)
                # Next token logits correspond to the last sequence position: [batch_size, vocab_size]
                next_token_logits = outputs.logits[:, -1, :].clone()

            # Step 3: Apply repetition penalty
            if repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, generated_ids, repetition_penalty
                )

            # Step 4: Apply decoding strategy
            strategy_lower = strategy.lower()

            if strategy_lower == 'greedy':
                # Deterministic argmax selection: argmax_w P(w | context)
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            elif strategy_lower == 'temperature':
                # Temperature scaled sampling
                scaled_logits = self._apply_temperature(next_token_logits, temperature)
                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            elif strategy_lower == 'top_k':
                # Temperature scaling followed by Top-K filtering
                scaled_logits = self._apply_temperature(next_token_logits, temperature)
                filtered_logits = self._top_k_filtering(scaled_logits, k=top_k)
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            elif strategy_lower == 'top_p':
                # Temperature scaling followed by Top-P (Nucleus) filtering
                scaled_logits = self._apply_temperature(next_token_logits, temperature)
                filtered_logits = self._top_p_filtering(scaled_logits, p=top_p)
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            else:
                raise ValueError(
                    f"Unknown decoding strategy '{strategy}'. "
                    f"Choose from: 'greedy', 'temperature', 'top_k', 'top_p'."
                )

            # Step 5: Check for End-Of-Sequence (EOS) token
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            # Step 6: Append chosen token to sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        # Step 7: Decode generated IDs back to human-readable string
        full_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return full_text

    def generate_with_details(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        strategy: str = 'top_p',
        repetition_penalty: float = 1.2
    ) -> List[Dict[str, Union[int, str, float, List[Tuple[str, float]]]]]:
        """Generate text and return step-by-step details for visualization.

        VIVA EXPLANATION:
        -----------------
        In a viva presentation or interactive UI, visualizing the step-by-step
        probabilities illustrates:
        - How the probability distribution shifts at every single token.
        - Which alternative candidates were in the Top-5 and what their probabilities were.
        - The model's degree of certainty at each point in the generation.

        Returns:
            list of dicts, each with:
            - 'step': 1-indexed generation step
            - 'token': the generated token string
            - 'token_id': the token id integer
            - 'probability': probability assigned to the chosen token
            - 'top_5': list of (token_str, prob) for the top 5 candidates at that step
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        generated_ids = input_ids.clone()
        step_details = []

        for step in range(1, max_new_tokens + 1):
            with torch.no_grad():
                outputs = self.model(generated_ids)
                next_token_logits = outputs.logits[:, -1, :].clone()

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, generated_ids, repetition_penalty
                )

            # Apply strategy-specific scaling & filtering
            strategy_lower = strategy.lower()

            if strategy_lower == 'greedy':
                # Probabilities under standard softmax for inspection
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            elif strategy_lower == 'temperature':
                scaled = self._apply_temperature(next_token_logits, temperature)
                probs = F.softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            elif strategy_lower == 'top_k':
                scaled = self._apply_temperature(next_token_logits, temperature)
                filtered = self._top_k_filtering(scaled, k=top_k)
                probs = F.softmax(filtered, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            elif strategy_lower == 'top_p':
                scaled = self._apply_temperature(next_token_logits, temperature)
                filtered = self._top_p_filtering(scaled, p=top_p)
                probs = F.softmax(filtered, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            else:
                raise ValueError(f"Unknown strategy: '{strategy}'")

            chosen_token_id = next_token.item()
            chosen_token_str = self.tokenizer.decode([chosen_token_id])
            chosen_prob = probs[0, chosen_token_id].item()

            # Extract top 5 candidates and their probabilities
            top5_probs, top5_indices = torch.topk(probs, 5, dim=-1)
            top_5_list = []
            for prob_val, idx_val in zip(top5_probs[0], top5_indices[0]):
                cand_str = self.tokenizer.decode([idx_val.item()])
                top_5_list.append((cand_str, round(prob_val.item(), 4)))

            step_details.append({
                'step': step,
                'token': chosen_token_str,
                'token_id': chosen_token_id,
                'probability': round(chosen_prob, 4),
                'top_5': top_5_list
            })

            # Check EOS
            if chosen_token_id == self.tokenizer.eos_token_id:
                break

            # Append token to sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        return step_details

    def generate_interview_question(
        self,
        role: str,
        difficulty: str,
        context: str = '',
        previous_qa: Optional[List[Tuple[str, str]]] = None,
        **kwargs
    ) -> str:
        """Generate a contextual follow-up interview question.

        Uses conditional text generation (Lab 4) where the generation
        is conditioned on the role, difficulty, and previous Q&A context.

        VIVA EXPLANATION:
        -----------------
        Conditional Language Modeling:
        Instead of unconditioned generation P(W), we model P(W | C) where
        C is the context representation. In autoregressive models like GPT-2,
        conditioning is achieved via 'prompt engineering' / prefix conditioning:
        the context C is prepended to the input sequence, and the model
        predicts tokens conditioned on that prefix.

        Args:
            role: Target job position (e.g., 'Python Backend Developer', 'Data Scientist')
            difficulty: Question difficulty (e.g., 'Junior', 'Mid-Level', 'Senior')
            context: Additional background or topic area (e.g., 'Concurrency, Memory Management')
            previous_qa: Optional list of (question, candidate_answer) pairs from earlier turns.
            **kwargs: Extra parameters passed to self.generate (temperature, top_p, etc.)

        Returns:
            The generated follow-up interview question string.
        """
        # Format previous conversation turns if provided
        history_str = ""
        if previous_qa:
            history_str = "\nPrevious Interview History:\n"
            for idx, (q, a) in enumerate(previous_qa, 1):
                history_str += f"Q{idx}: {q}\nA{idx}: {a}\n"

        # Build a structured conditioning prompt
        structured_prompt = (
            f"The following is a {difficulty} technical interview for a {role} position.\n"
            f"{context.strip()}\n"
            f"{history_str.strip()}\n\n"
            f"Interviewer's next question:"
        )

        full_output = self.generate(structured_prompt, **kwargs)

        # Extract only the newly generated text following the prompt
        if "Interviewer's next question:" in full_output:
            question_part = full_output.split("Interviewer's next question:")[-1].strip()
            # If the model starts hallucinating multiple turns, extract just the first question
            first_question = question_part.split("\n")[0].strip()
            return first_question if first_question else question_part
        
        return full_output.strip()


# =====================================================================
# Standalone Demonstration & Viva Explanation Execution Block
# =====================================================================
if __name__ == '__main__':
    print("=" * 75)
    print("SMARTHIRE: Autoregressive Text Generation & Decoding Algorithms")
    print("Lab 3 & 4: Custom Decoding Demonstration for Viva Examination")
    print("=" * 75)
    
    # Initialize generator
    generator = InterviewGenerator(model_name='distilgpt2')

    test_prompt = "In Python, the primary difference between a list and a tuple is"
    print(f"Sample Prompt:\n\"{test_prompt}\"\n")
    print("-" * 75)

    # 1. Greedy Search Demonstration
    print("1. GREEDY DECODING (strategy='greedy')")
    print("   VIVA CONCEPT: At each step t, selects w_t = argmax P(w | w_{<t}).")
    print("   Pros: Deterministic, fast. Cons: Suffers from repetitive loops and myopic choices.")
    greedy_output = generator.generate(
        test_prompt,
        max_new_tokens=40,
        strategy='greedy',
        repetition_penalty=1.2
    )
    print(f"   Output:\n   {greedy_output}\n")
    print("-" * 75)

    # 2. Temperature Sampling Demonstration
    print("2. TEMPERATURE SAMPLING (strategy='temperature', temperature=1.2)")
    print("   VIVA CONCEPT: Divides logits by T before softmax. T > 1 increases diversity.")
    print("   Pros: Diverse, creative. Cons: Risk of sampling rare/nonsensical tokens.")
    temp_output = generator.generate(
        test_prompt,
        max_new_tokens=40,
        temperature=1.2,
        strategy='temperature',
        repetition_penalty=1.2
    )
    print(f"   Output:\n   {temp_output}\n")
    print("-" * 75)

    # 3. Top-K Sampling Demonstration
    print("3. TOP-K SAMPLING (strategy='top_k', top_k=20, temperature=0.7)")
    print("   VIVA CONCEPT: Restricts sampling pool to top K candidates; truncates tail.")
    print("   Pros: Eliminates bizarre tokens. Cons: Fixed K does not adapt to certainty.")
    topk_output = generator.generate(
        test_prompt,
        max_new_tokens=40,
        top_k=20,
        temperature=0.7,
        strategy='top_k',
        repetition_penalty=1.2
    )
    print(f"   Output:\n   {topk_output}\n")
    print("-" * 75)

    # 4. Top-P (Nucleus) Sampling Demonstration
    print("4. TOP-P / NUCLEUS SAMPLING (strategy='top_p', top_p=0.85, temperature=0.7)")
    print("   VIVA CONCEPT: Dynamically adapts pool size to cumulative probability mass p.")
    print("   Pros: Balances diversity and coherence by adapting to model confidence.")
    topp_output = generator.generate(
        test_prompt,
        max_new_tokens=40,
        top_p=0.85,
        temperature=0.7,
        strategy='top_p',
        repetition_penalty=1.2
    )
    print(f"   Output:\n   {topp_output}\n")
    print("-" * 75)

    # 5. Step-by-Step Probability Inspection (generate_with_details)
    print("5. STEP-BY-STEP PROBABILITY TRACE (generate_with_details)")
    print("   Inspects top-5 token distribution for first 5 generated steps:")
    details = generator.generate_with_details(
        prompt="A thread in an operating system is",
        max_new_tokens=5,
        top_p=0.9,
        temperature=0.7,
        strategy='top_p'
    )
    for item in details:
        print(f"   Step {item['step']}: Chosen='{item['token'].strip()}' (p={item['probability']:.4f})")
        print(f"           Top 5 candidates: {item['top_5']}")
    print("-" * 75)

    # 6. Conditional Interview Question Generation
    print("6. CONDITIONAL INTERVIEW QUESTION GENERATION")
    question = generator.generate_interview_question(
        role="Backend Python Engineer",
        difficulty="Mid-Level",
        context="Candidate has experience with FastAPI, PostgreSQL, and Redis caching.",
        previous_qa=[
            ("How do you handle database connection pooling in Python?",
             "I use SQLAlchemy with QueuePool to manage and reuse connections.")
        ],
        max_new_tokens=50,
        temperature=0.8,
        strategy='top_p'
    )
    print(f"   Generated Question:\n   \"{question}\"")
    print("=" * 75)
