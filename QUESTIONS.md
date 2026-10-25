# Questions

What one GPT-2 decode step asks. What, never how.

One step takes one token `x` at position `t`. It answers 10 questions.
Questions 2 to 8 are asked once per layer, for 12 layers.

Every answer is defined by the Hugging Face GPT-2 model (`gpt2`, fp32).
The HF names in each "Defined by" line say which value counts as right.

Sizes: width 768, 12 heads of 64, MLP width 3072, vocabulary 50257.

---

## Q1. What does this token mean, here?

Given the token `x` and its position `t`.
Answer: the word's meaning plus the position's meaning, added together.

Defined by: `wte[x] + wpe[t]`. Equals HF `hidden_states[0]` at `t`.

## Q2. What is the state, on a common scale?

Given the current state.
Answer: the state with mean 0 and spread 1, then scaled and shifted by the layer's learned values.

Defined by: `h.{l}.ln_1`, epsilon 1e-5.

## Q3. What does this token ask, offer, and carry?

Given the scaled state from Q2.
Answer: three things per head.
- A query: what it is looking for.
- A key: what it can be found by.
- A value: what it hands over if found.

Defined by: `h.{l}.attn.c_attn`, split into query, key, value, then into 12 heads.

## Q4. What has the model seen so far?

Given every earlier answer to Q3 in this layer, and this one.
Answer: the keys and values for positions 0 to `t`. Nothing after `t`.

Defined by: HF `past_key_values` for layer `l`, extended by this step's key and value.

## Q5. How well does each past position fit the query?

Given this token's query, and the keys from Q4.
Answer: one score per head per position `s` from 0 to `t`. The score is query dot key, divided by 8 (the square root of 64).

Defined by: HF attention scores for layer `l` before softmax, causal.

## Q6. How much attention does each past position get?

Given the scores from Q5.
Answer: per head, weights that are all positive and add to 1. A higher score gets more weight.

Defined by: softmax over `s` from 0 to `t`. Equals HF `attentions[l]` at row `t`.

## Q7. What does the past say back?

Given the weights from Q6 and the values from Q4.
Answer: per head, the weighted sum of values. The 12 heads are joined, projected, and added to the state.

Defined by: `h.{l}.attn.c_proj`, then the residual add.

## Q8. What does the token make of it?

Given the state after Q7.
Answer: scale it (as Q2, with this layer's second set of values). Widen to 3072, apply GELU, narrow back to 768. Add to the state.

Defined by: `h.{l}.ln_2`, `h.{l}.mlp` (GELU tanh form), then the residual add.
After layer 11 this equals HF `hidden_states[12]` before `ln_f`.

## Q9. How likely is each next word?

Given the state after all 12 layers.
Answer: scale it (final values), then one score per word in the vocabulary.
The word table from Q1 is reused here.

Defined by: `ln_f`, then `wte` transposed. Equals HF `logits` at `t`.

## Q10. Which word comes next?

Given the scores from Q9.
Answer: the word with the highest score. On a tie, the lowest word id.

Defined by: `argmax`, matching HF greedy `generate`.

---

## What is not asked

- How the work is split, ordered, or placed.
- Where any value lives between questions.
- Anything about batches. A batch is the same 10 questions asked of several tokens at once.
