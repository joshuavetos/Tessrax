# TMP-1 Prompt: Conversation Contradiction Sweep  
**Version:** 1.0  
**Purpose:** Run the Tessrax Minimal Protocol (TMP-1) against any conversation or text to detect, classify, and metabolize contradictions.

---

## 🧠 Protocol

### TMP-1 Loop
1. **OBSERVE** – Parse each message into discrete claims  
2. **DETECT** – Identify incompatible or negated pairs  
3. **EVALUATE** – Compute stability = 1 − (|contradictions| ÷ |claims|)  
4. **ROUTE** –  
   • > 0.8 → `accept`  
   • 0.5 – 0.8 → `reconcile`  
   • < 0.5 → `reset`  
5. **ACT** – Suggest reconciliations or resets  
6. **RECORD** – Output findings as structured JSON  

---

## 📋 Input
Paste a full conversation transcript or text corpus.

---

## 🧩 Output Schema
```json
{
  "claims_analyzed": 0,
  "contradictions_found": [
    {
      "type": "Logical | Temporal | Normative | Semantic | Procedural",
      "pair": ["Claim A", "Claim B"],
      "severity": 0.0,
      "explanation": "",
      "possible_reconciliation": ""
    }
  ],
  "stability_score": 0.00,
  "recommended_route": "accept | reconcile | reset",
  "summary": "",
  "reflection": "What this contradiction means for alignment, coherence, or next actions."
}


⸻

🧮 Severity Scale

Range	Meaning
0.0 – 0.3	Minor variance
0.3 – 0.6	Moderate contradiction
0.6 – 1.0	Severe / systemic contradiction


⸻

🧭 Example Prompt

Perform a TMP-1 contradiction sweep on the following transcript.
Detect Logical, Temporal, Normative, Semantic, and Procedural contradictions.
Return results in the JSON schema above and include a governance reflection.

⸻

🔍 Example Output

{
  "claims_analyzed": 47,
  "contradictions_found": [
    {
      "type": "Normative",
      "pair": ["I trust this system", "Nothing is reliable"],
      "severity": 0.8,
      "explanation": "Opposing stances on reliability",
      "possible_reconciliation": "Different timeframes—trust in design vs. frustration in use."
    }
  ],
  "stability_score": 0.57,
  "recommended_route": "reconcile",
  "summary": "Conversation alternated between confidence and disillusionment.",
  "reflection": "Visible tension between ideals and lived experience—requires contextual reconciliation."
}


⸻

🧩 Usage

Use in any agent (Claude, Gemini, Perplexity, GPT):

Run a Tessrax TMP-1 contradiction sweep on this conversation.
Return JSON + reflection per specification.


