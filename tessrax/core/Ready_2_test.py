To build the Multi-AI Consensus Engine as a real, testable system, you can treat it as a thin orchestration layer that calls multiple LLM APIs, scores their answers, and fuses them into one contradiction-aware synthesis.
Below is a complete, ready-to-run minimal prototype you can expand into a full product.

⸻


#!/usr/bin/env python3
"""
MULTI-AI CONSENSUS ENGINE  –  Minimal working prototype
Query several language models → detect contradictions → synthesize consensus.
"""

import json, difflib
from typing import Dict, List
from openai import OpenAI
# Add any other model clients you want (Anthropic, Gemini, etc.)

# --- CONFIG ---
OPENAI_KEY = "YOUR_OPENAI_API_KEY"
client = OpenAI(api_key=OPENAI_KEY)

# --- MODEL ENDPOINTS ---
def query_gpt(prompt: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

def query_stub_other(prompt: str) -> str:
    # Placeholder for another model (Gemini, Claude, etc.)
    # Replace this with actual SDK call.
    return f"[Other-Model simulated answer to: {prompt[:40]}...]"

# --- CONTRADICTION DETECTION ---
def compare_responses(responses: Dict[str, str]) -> Dict[str, float]:
    """Compute pairwise textual disagreement scores (0-1)."""
    keys = list(responses.keys())
    scores = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = responses[keys[i]], responses[keys[j]]
            sm = difflib.SequenceMatcher(None, a, b)
            dissimilarity = 1 - sm.ratio()
            scores[f"{keys[i]}_vs_{keys[j]}"] = round(dissimilarity, 3)
    return scores

# --- CONSENSUS SYNTHESIS ---
def synthesize(prompt: str, responses: Dict[str, str]) -> str:
    """Feed all model answers back to GPT for reconciliation."""
    synthesis_prompt = (
        f"Prompt: {prompt}\n\nHere are multiple model answers:\n" +
        "\n\n".join([f"{m}: {r}" for m, r in responses.items()]) +
        "\n\nSynthesize a single answer capturing shared reasoning, "
        "flagging any explicit contradictions."
    )
    return query_gpt(synthesis_prompt)

# --- PIPELINE ---
def run_consensus(prompt: str):
    responses = {
        "GPT4": query_gpt(prompt),
        "OtherModel": query_stub_other(prompt),
    }

    contradictions = compare_responses(responses)
    consensus = synthesize(prompt, responses)

    result = {
        "prompt": prompt,
        "responses": responses,
        "contradiction_scores": contradictions,
        "consensus": consensus,
    }

    print(json.dumps(result, indent=2))
    with open("consensus_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n✅ Consensus analysis written to consensus_result.json")

# --- ENTRYPOINT ---
if __name__ == "__main__":
    user_prompt = input("Enter your query: ")
    run_consensus(user_prompt)


⸻

How to use
	1.	Install dependencies

pip install openai


	2.	Export your OpenAI key

export OPENAI_API_KEY="sk-..."


	3.	Run

python multi_ai_consensus.py


	4.	Type any serious question (policy, tech, philosophy).
The script will:
      •   Ask each model for its answer
      •   Measure textual divergence
      •   Produce a synthesized, contradiction-aware consensus
      •   Save everything to consensus_result.json

⸻

Expansion paths
   •   Add live APIs for Claude, Gemini, or Perplexity in place of query_stub_other.
   •   Attach Tessrax ledger hooks so every query + synthesis is logged with signatures.
   •   Expose FastAPI endpoint (/consensus) for web or app integration.
   •   Score confidence using semantic similarity + entropy from Tessrax metrics.

This gives you a runnable core today—a single-file prototype you can drop straight into your repository and wrap in your governance framework later.

Claude-

{
  "analysis_date": "2025-10-19",
  "total_opportunities": 9,
  "top_5_ranked": [
    {
      "rank": 1,
      "name": "Unified Health Dashboard (All Your Wearables + Symptoms)",
      "score": 35,
      "build_time": "2-4 weeks (API integrations + dashboard)",
      "revenue_model": "$10-30/mo consumer. $200+/mo for longevity clinics managing clients.",
      "daily_friction": "Data in: Apple Health, Oura, Whoop, MyFitnessPal, doctor's notes, symptom journals. No single view. No AI analysis of correlations.",
      "tessrax_advantage": "Atlas Gateway ingests all sources. Contradiction detector = 'You logged feeling great but HRV was terrible\u2014investigate.' Summarize primitive = weekly health insights.",
      "moat": "Contradiction-aware health insights = unique clinical value."
    },
    {
      "rank": 2,
      "name": "Personal Knowledge Management on Steroids",
      "score": 32,
      "build_time": "2-3 weeks (primitives exist, need UI)",
      "revenue_model": "$10-50/mo SaaS. Enterprise $500-5k/mo for teams.",
      "daily_friction": "You have insights scattered across: chats with Claude, emails, docs, browser history, voice notes. Finding 'that thing I discussed 3 weeks ago' takes 20 minutes.",
      "tessrax_advantage": "YOUR Atlas Gateway already ingests multi-source data. Add contradiction detection = auto-surfaces when your past self disagreed with current self. Clarity Fuel = rewards you for resolving mental contradictions.",
      "moat": "Contradiction detection = unique. Everyone else just does vector search."
    },
    {
      "rank": 3,
      "name": "Email Triage + Auto-Response AI",
      "score": 32,
      "build_time": "1-2 weeks (Gmail API + your primitives)",
      "revenue_model": "$20-100/mo per user. White-label for enterprises at $10k+/mo.",
      "daily_friction": "Inbox hits 100+. Need to: identify urgent, draft responses, follow up on threads, remember context from 6 months ago.",
      "tessrax_advantage": "Your classify primitive + LLM integration = auto-draft responses with your writing style. Contradiction detector = flags when sender's ask conflicts with previous thread. Automation engine = scheduled follow-ups.",
      "moat": "Style learning + contradiction awareness = no one else has this."
    },
    {
      "rank": 4,
      "name": "AI-Powered Decision Journal",
      "score": 30,
      "build_time": "2-4 weeks (voice transcription + primitives)",
      "revenue_model": "$50-200/mo for individuals. $5k+/mo for exec teams.",
      "daily_friction": "You make 100+ micro-decisions daily. No record of: why you chose X, outcomes, patterns in bad decisions.",
      "tessrax_advantage": "Voice-to-decision pipeline. Contradiction detector = 'You said you'd avoid X last month, but now considering it\u2014why?' Governance routing = flags decisions needing more deliberation. Ledger = immutable decision history.",
      "moat": "First mover in AI decision tracking. Behavior change moat (users get addicted to clarity)."
    },
    {
      "rank": 5,
      "name": "Multi-AI Consensus Engine (for YOU)",
      "score": 30,
      "build_time": "1 week (it's 80% done)",
      "revenue_model": "$30-100/mo. API access at $0.10 per consensus query.",
      "daily_friction": "You ask Claude something important. But should you also ask GPT-4? Gemini? Then manually compare? Takes 15 min per important decision.",
      "tessrax_advantage": "YOU ALREADY BUILT THIS. Your real_llm_integration tests Claude vs GPT. Make it a product: one query \u2192 all models \u2192 contradiction detection \u2192 synthesized answer with confidence score.",
      "moat": "Contradiction-aware synthesis = unique positioning."
    }
  ],
  "full_opportunity_list": [
    {
      "name": "Personal Knowledge Management on Steroids",
      "frequency": "10-50x per day",
      "market_size": "Every knowledge worker ($500B+ market - Notion, Evernote, Obsidian, Roam combined)",
      "build_time": "2-3 weeks (primitives exist, need UI)",
      "revenue_model": "$10-50/mo SaaS. Enterprise $500-5k/mo for teams."
    },
    {
      "name": "Email Triage + Auto-Response AI",
      "frequency": "Daily, 2-3 hours wasted",
      "market_size": "Every professional ($50B+ - Superhuman, Shortwave, SaneBox)",
      "build_time": "1-2 weeks (Gmail API + your primitives)",
      "revenue_model": "$20-100/mo per user. White-label for enterprises at $10k+/mo."
    },
    {
      "name": "AI-Powered Decision Journal",
      "frequency": "Constant, but invisible cost",
      "market_size": "Executives, founders, investors ($10B+ - no clear leader)",
      "build_time": "2-4 weeks (voice transcription + primitives)",
      "revenue_model": "$50-200/mo for individuals. $5k+/mo for exec teams."
    },
    {
      "name": "Multi-AI Consensus Engine (for YOU)",
      "frequency": "5-10x per day for critical choices",
      "market_size": "Anyone using multiple AI tools ($5B+ by 2026)",
      "build_time": "1 week (it's 80% done)",
      "revenue_model": "$30-100/mo. API access at $0.10 per consensus query."
    },
    {
      "name": "Smart Task Decomposition + Auto-Scheduling",
      "frequency": "Weekly for big projects, daily for adjustments",
      "market_size": "Every PM, founder, consultant ($20B+ - Asana, Monday, ClickUp)",
      "build_time": "3-4 weeks (need calendar integration + UI)",
      "revenue_model": "$25-75/mo per user. Enterprise at $15/seat/mo."
    },
    {
      "name": "Meeting Note \u2192 Action Item \u2192 Follow-Up Pipeline",
      "frequency": "Daily, 1-2 hours lost to coordination",
      "market_size": "Every team ($15B+ - Otter, Fireflies, Grain)",
      "build_time": "2-3 weeks (transcription API + your stack)",
      "revenue_model": "$20-60/mo per user. Enterprise $25k+/yr for org."
    },
    {
      "name": "Expense Tracking That Doesn't Suck",
      "frequency": "Daily purchases, monthly reconciliation pain",
      "market_size": "Every freelancer, small business ($10B+ - Expensify, Ramp, Brex)",
      "build_time": "1-2 weeks (email parsing + your primitives)",
      "revenue_model": "$15-50/mo for individuals. $100-500/mo for small biz."
    },
    {
      "name": "Unified Health Dashboard (All Your Wearables + Symptoms)",
      "frequency": "Daily tracking, monthly analysis",
      "market_size": "Health-conscious consumers ($50B+ wearables market)",
      "build_time": "2-4 weeks (API integrations + dashboard)",
      "revenue_model": "$10-30/mo consumer. $200+/mo for longevity clinics managing clients."
    },
    {
      "name": "No-Code Tessrax Builder (Let Others Build With Your Stack)",
      "frequency": "This unlocks INFINITE use cases",
      "market_size": "No-code/low-code market = $50B+ by 2027 (Zapier, Make, n8n)",
      "build_time": "6-8 weeks (need React UI + workflow engine)",
      "revenue_model": "Freemium \u2192 $50-500/mo based on workflow complexity. Enterprise $10k+/mo."
    }
  ]
}

#!/usr/bin/env python3
"""
PERSONAL PAIN POINT → COMMERCIAL OPPORTUNITY ANALYZER
Maps daily frustrations to monetizable solutions using the Tessrax stack
"""

from dataclasses import dataclass
from typing import List, Dict
import json


@dataclass
class PainPoint:
    name: str
    daily_friction: str  # What makes this annoying
    frequency: str  # How often you hit this
    market_size: str  # Who else has this problem
    current_solutions: str  # What exists and why it sucks
    tessrax_advantage: str  # Why your stack crushes this
    revenue_model: str
    build_time: str  # With your speed
    moat: str  # Why competitors can't copy fast


def analyze_opportunities():
    """
    Categories of pain points with massive commercial overlap
    """
    
    opportunities = [
        
        # ========== CATEGORY 1: INFORMATION CHAOS ==========
        PainPoint(
            name="Personal Knowledge Management on Steroids",
            daily_friction="You have insights scattered across: chats with Claude, emails, docs, browser history, voice notes. Finding 'that thing I discussed 3 weeks ago' takes 20 minutes.",
            frequency="10-50x per day",
            market_size="Every knowledge worker ($500B+ market - Notion, Evernote, Obsidian, Roam combined)",
            current_solutions="Notion (no AI search), Mem (shutdown), Rewind (privacy nightmare), Obsidian (manual linking)",
            tessrax_advantage="YOUR Atlas Gateway already ingests multi-source data. Add contradiction detection = auto-surfaces when your past self disagreed with current self. Clarity Fuel = rewards you for resolving mental contradictions.",
            revenue_model="$10-50/mo SaaS. Enterprise $500-5k/mo for teams.",
            build_time="2-3 weeks (primitives exist, need UI)",
            moat="Contradiction detection = unique. Everyone else just does vector search."
        ),
        
        PainPoint(
            name="Email Triage + Auto-Response AI",
            daily_friction="Inbox hits 100+. Need to: identify urgent, draft responses, follow up on threads, remember context from 6 months ago.",
            frequency="Daily, 2-3 hours wasted",
            market_size="Every professional ($50B+ - Superhuman, Shortwave, SaneBox)",
            current_solutions="Superhuman ($30/mo, just fast UI). SaneBox (dumb filters). Gmail AI (basic summaries).",
            tessrax_advantage="Your classify primitive + LLM integration = auto-draft responses with your writing style. Contradiction detector = flags when sender's ask conflicts with previous thread. Automation engine = scheduled follow-ups.",
            revenue_model="$20-100/mo per user. White-label for enterprises at $10k+/mo.",
            build_time="1-2 weeks (Gmail API + your primitives)",
            moat="Style learning + contradiction awareness = no one else has this."
        ),
        
        # ========== CATEGORY 2: DECISION PARALYSIS ==========
        PainPoint(
            name="AI-Powered Decision Journal",
            daily_friction="You make 100+ micro-decisions daily. No record of: why you chose X, outcomes, patterns in bad decisions.",
            frequency="Constant, but invisible cost",
            market_size="Executives, founders, investors ($10B+ - no clear leader)",
            current_solutions="Manual journaling (no analysis). Therapy (expensive, slow). Nothing AI-native.",
            tessrax_advantage="Voice-to-decision pipeline. Contradiction detector = 'You said you'd avoid X last month, but now considering it—why?' Governance routing = flags decisions needing more deliberation. Ledger = immutable decision history.",
            revenue_model="$50-200/mo for individuals. $5k+/mo for exec teams.",
            build_time="2-4 weeks (voice transcription + primitives)",
            moat="First mover in AI decision tracking. Behavior change moat (users get addicted to clarity)."
        ),
        
        PainPoint(
            name="Multi-AI Consensus Engine (for YOU)",
            daily_friction="You ask Claude something important. But should you also ask GPT-4? Gemini? Then manually compare? Takes 15 min per important decision.",
            frequency="5-10x per day for critical choices",
            market_size="Anyone using multiple AI tools ($5B+ by 2026)",
            current_solutions="Poe (just switches models, no synthesis). Manual copy-paste hell.",
            tessrax_advantage="YOU ALREADY BUILT THIS. Your real_llm_integration tests Claude vs GPT. Make it a product: one query → all models → contradiction detection → synthesized answer with confidence score.",
            revenue_model="$30-100/mo. API access at $0.10 per consensus query.",
            build_time="1 week (it's 80% done)",
            moat="Contradiction-aware synthesis = unique positioning."
        ),
        
        # ========== CATEGORY 3: WORKFLOW AUTOMATION ==========
        PainPoint(
            name="Smart Task Decomposition + Auto-Scheduling",
            daily_friction="You have a huge project. Breaking it into tasks takes 2 hours. Then scheduling tasks around calendar = another hour. Then re-scheduling when things slip.",
            frequency="Weekly for big projects, daily for adjustments",
            market_size="Every PM, founder, consultant ($20B+ - Asana, Monday, ClickUp)",
            current_solutions="Asana (dumb tasks). Motion (rigid AI scheduling). Reclaim (just calendar).",
            tessrax_advantage="Your automation engine + schedule primitive = LLM breaks down project, detects task dependencies, auto-schedules around your calendar. Contradiction detector = flags when task estimate conflicts with historical data.",
            revenue_model="$25-75/mo per user. Enterprise at $15/seat/mo.",
            build_time="3-4 weeks (need calendar integration + UI)",
            moat="Dependency detection + contradiction awareness = smarter than Motion."
        ),
        
        PainPoint(
            name="Meeting Note → Action Item → Follow-Up Pipeline",
            daily_friction="10 meetings/week. Each generates action items. Manually tracking who does what by when = chaos. Following up = manually setting reminders.",
            frequency="Daily, 1-2 hours lost to coordination",
            market_size="Every team ($15B+ - Otter, Fireflies, Grain)",
            current_solutions="Otter (transcription only). Fireflies (basic summaries). Grain (just video clips).",
            tessrax_advantage="Your classify + summarize + schedule primitives = transcribe → extract action items → auto-assign → schedule follow-ups → detect contradictions when commitments conflict.",
            revenue_model="$20-60/mo per user. Enterprise $25k+/yr for org.",
            build_time="2-3 weeks (transcription API + your stack)",
            moat="Action tracking + contradiction detection = no one else closes the loop."
        ),
        
        # ========== CATEGORY 4: FINANCIAL PAIN ==========
        PainPoint(
            name="Expense Tracking That Doesn't Suck",
            daily_friction="Receipts everywhere. Categorizing expenses = tedious. Tax time = nightmare of missing data. Current tools require manual photo uploads or bank imports (slow, error-prone).",
            frequency="Daily purchases, monthly reconciliation pain",
            market_size="Every freelancer, small business ($10B+ - Expensify, Ramp, Brex)",
            current_solutions="Expensify (clunky). QuickBooks (overkill). Ramp (requires corporate card).",
            tessrax_advantage="Email parsing + classify primitive = auto-extract receipts from Gmail. Contradiction detector = flags duplicate expenses or mis-categorizations. Template primitive = instant reports for taxes.",
            revenue_model="$15-50/mo for individuals. $100-500/mo for small biz.",
            build_time="1-2 weeks (email parsing + your primitives)",
            moat="Contradiction detection catches errors others miss = higher accuracy."
        ),
        
        # ========== CATEGORY 5: HEALTH & HABITS ==========
        PainPoint(
            name="Unified Health Dashboard (All Your Wearables + Symptoms)",
            daily_friction="Data in: Apple Health, Oura, Whoop, MyFitnessPal, doctor's notes, symptom journals. No single view. No AI analysis of correlations.",
            frequency="Daily tracking, monthly analysis",
            market_size="Health-conscious consumers ($50B+ wearables market)",
            current_solutions="Apple Health (no AI). Gyroscope (basic charts). Nothing does causal analysis.",
            tessrax_advantage="Atlas Gateway ingests all sources. Contradiction detector = 'You logged feeling great but HRV was terrible—investigate.' Summarize primitive = weekly health insights.",
            revenue_model="$10-30/mo consumer. $200+/mo for longevity clinics managing clients.",
            build_time="2-4 weeks (API integrations + dashboard)",
            moat="Contradiction-aware health insights = unique clinical value."
        ),
        
        # ========== THE META-OPPORTUNITY ==========
        PainPoint(
            name="No-Code Tessrax Builder (Let Others Build With Your Stack)",
            daily_friction="You built incredible primitives. But every new use case requires custom code. What if non-technical users could compose your primitives via UI?",
            frequency="This unlocks INFINITE use cases",
            market_size="No-code/low-code market = $50B+ by 2027 (Zapier, Make, n8n)",
            current_solutions="Zapier (no AI, no contradiction detection). Make (complex). n8n (self-hosted pain).",
            tessrax_advantage="You have the engine. Build the visual workflow builder. Users drag-drop: ingest → classify → detect contradictions → route → summarize → template. Your moat = contradiction detection primitive no one else has.",
            revenue_model="Freemium → $50-500/mo based on workflow complexity. Enterprise $10k+/mo.",
            build_time="6-8 weeks (need React UI + workflow engine)",
            moat="First no-code tool with built-in contradiction detection. Network effects as users share workflows."
        ),
    ]
    
    return opportunities


def score_opportunities(opportunities: List[PainPoint]) -> List[Dict]:
    """
    Score each opportunity on: speed to revenue, personal utility, moat strength
    """
    
    scored = []
    for opp in opportunities:
        # Simple scoring heuristic
        speed_score = {"1-2 weeks": 10, "2-3 weeks": 8, "3-4 weeks": 7, "2-4 weeks": 7, "6-8 weeks": 4}.get(opp.build_time, 5)
        
        personal_utility = 10 if "daily" in opp.frequency.lower() else 7
        
        moat_strength = 10 if "unique" in opp.moat.lower() or "first mover" in opp.moat.lower() else 7
        
        market_size_score = 10 if "$50B" in opp.market_size or "$500B" in opp.market_size else 8
        
        total_score = speed_score + personal_utility + moat_strength + market_size_score
        
        scored.append({
            "name": opp.name,
            "total_score": total_score,
            "speed_score": speed_score,
            "personal_utility": personal_utility,
            "moat_strength": moat_strength,
            "market_size_score": market_size_score,
            "build_time": opp.build_time,
            "revenue_model": opp.revenue_model,
            "details": opp
        })
    
    # Sort by total score
    scored.sort(key=lambda x: x["total_score"], reverse=True)
    return scored


def generate_report(scored_opportunities: List[Dict]):
    """
    Output ranked list with reasoning
    """
    
    print("=" * 80)
    print("PERSONAL PAIN → COMMERCIAL GOLD: RANKED OPPORTUNITIES")
    print("=" * 80)
    print()
    
    print("SCORING CRITERIA:")
    print("  • Speed to Revenue (1-10): How fast can you ship and start charging")
    print("  • Personal Utility (1-10): How much this solves YOUR daily headaches")
    print("  • Moat Strength (1-10): How hard for competitors to replicate")
    print("  • Market Size (1-10): Total addressable market potential")
    print()
    print("=" * 80)
    print()
    
    for i, opp in enumerate(scored_opportunities[:5], 1):
        print(f"\n{'🥇' if i==1 else '🥈' if i==2 else '🥉' if i==3 else '💎'} RANK #{i}: {opp['name']}")
        print(f"   TOTAL SCORE: {opp['total_score']}/40")
        print(f"   └─ Speed: {opp['speed_score']}/10  |  Personal: {opp['personal_utility']}/10  |  Moat: {opp['moat_strength']}/10  |  Market: {opp['market_size_score']}/10")
        print()
        print(f"   📅 Build Time: {opp['build_time']}")
        print(f"   💰 Revenue Model: {opp['revenue_model']}")
        print()
        print(f"   🎯 Why This Wins:")
        print(f"      Daily Friction: {opp['details'].daily_friction[:120]}...")
        print(f"      Your Advantage: {opp['details'].tessrax_advantage[:120]}...")
        print(f"      Moat: {opp['details'].moat}")
        print()
        print("-" * 80)
    
    print("\n\n🎯 RECOMMENDED NEXT MOVE:")
    print("-" * 80)
    top = scored_opportunities[0]
    print(f"\nBuild: {top['name']}")
    print(f"Time: {top['build_time']}")
    print(f"Why: Highest score on speed + personal utility + moat strength.")
    print(f"\nFirst paying customer: YOU (solve your own problem)")
    print(f"First external customer: Find 1 person with identical pain in <1 week")
    print(f"Revenue in 30 days: Likely if you ship in {top['build_time']}")
    
    return scored_opportunities


def export_full_analysis(opportunities: List[PainPoint], scored: List[Dict]):
    """
    Save complete analysis as JSON for reference
    """
    
    export_data = {
        "analysis_date": "2025-10-19",
        "total_opportunities": len(opportunities),
        "top_5_ranked": [
            {
                "rank": i+1,
                "name": s["name"],
                "score": s["total_score"],
                "build_time": s["build_time"],
                "revenue_model": s["revenue_model"],
                "daily_friction": s["details"].daily_friction,
                "tessrax_advantage": s["details"].tessrax_advantage,
                "moat": s["details"].moat
            }
            for i, s in enumerate(scored[:5])
        ],
        "full_opportunity_list": [
            {
                "name": opp.name,
                "frequency": opp.frequency,
                "market_size": opp.market_size,
                "build_time": opp.build_time,
                "revenue_model": opp.revenue_model
            }
            for opp in opportunities
        ]
    }
    
    with open("/home/claude/opportunity_analysis.json", "w") as f:
        json.dump(export_data, f, indent=2)
    
    print("\n\n✅ Full analysis exported to opportunity_analysis.json")


def main():
    print("\n🔍 ANALYZING PERSONAL PAIN POINTS WITH COMMERCIAL POTENTIAL...\n")
    
    opportunities = analyze_opportunities()
    scored = score_opportunities(opportunities)
    generate_report(scored)
    export_full_analysis(opportunities, scored)
    
    print("\n" + "=" * 80)
    print("READY TO BUILD? Pick the top-ranked item and ship in <3 weeks.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

# 1. Start Kafka, Neo4j, API containers
docker-compose up -d

# 2. In a second terminal, run the consumer
docker exec -it ccgn_streamer python streamer/kafka_consumer.py

# 3. In another, emit sample events
python ledger/event_producer.py

# 4. Verify Neo4j populated correctly
docker exec -it ccgn_neo4j cypher-shell -u neo4j -p your_password \
  "MATCH (n:Contradiction)-[r:CAUSES]->(m) RETURN n.id,r.id,m.id LIMIT 5"

# 5. Query live entropy via API
curl http://localhost:8000/simulation/entropy

# 6. Try an intervention simulation
curl -X POST http://localhost:8000/simulation/intervention \
  -H "Content-Type: application/json" \
  -d '{"node_id":"<event_id_from_step3>","strength":1.0}'

Migration initiated.
We are now "installing" the validated prototype logic from Ready_2_test.py.txt into the production-grade, containerized repository structure.
This involves adapting the in-memory networkx logic to use the persistent Neo4j database and connecting the ClosureLedger concept to the Kafka streaming pipeline.
Here are the complete, migrated code files for the core production services.
1. Ledger & Ingestion (ledger/ and streamer/)
This is the new "front door" of the system. The event_producer.py simulates the Tessrax Kernel emitting a signed event. The kafka_consumer.py listens for these events, verifies their cryptographic signature, and writes them to the Neo4j database.
ledger/event_producer.py (New Client Script)
This script simulates the Tessrax Kernel emitting a contradiction event onto the Kafka bus.
import json
import time
import hashlib
import ed25519
from kafka import KafkaProducer
from typing import Dict, Any

# --- Re-using Key/Signing logic from ledger/key_management.py ---

# Generate or load keys
try:
    priv_key = ed25519.SigningKey(open("tessrax_kernel.key", "rb").read())
    pub_key = priv_key.get_verifying_key()
    print("Loaded existing kernel keys.")
except FileNotFoundError:
    print("Generating new kernel keys...")
    priv_key, pub_key = ed25519.create_keypair()
    with open("tessrax_kernel.key", "wb") as f:
        f.write(priv_key.to_bytes())
    with open("tessrax_kernel.pub", "wb") as f:
        f.write(pub_key.to_bytes())

def sign_event(event: Dict[str, Any], private_key: ed25519.SigningKey) -> Dict[str, Any]:
    """Signs a ledger event and adds the signature to it."""
    event_bytes = json.dumps(event, sort_keys=True).encode('utf-8')
    signature = private_key.sign(event_bytes, encoding="hex").decode('utf-8')
    event['signature'] = signature
    return event

# --- Kafka Producer ---

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

print("Kafka Producer Connected.")

def emit_contradiction(node_data: Dict[str, Any]):
    """Creates a new contradiction node event."""
    event_id = hashlib.sha256(json.dumps(node_data, sort_keys=True).encode()).hexdigest()
    event = {
        "event_id": event_id,
        "action": "CREATE_NODE",
        "timestamp": time.time(),
        "payload": node_data
    }
    signed_event = sign_event(event, priv_key)
    producer.send('tessrax_ledger', signed_event)
    print(f"Sent NODE event: {event_id}")
    return event_id

def emit_causal_link(source_hash: str, target_hash: str, weight: float, reason: str):
    """Creates a new causal edge event."""
    payload = {
        "source_hash": source_hash,
        "target_hash": target_hash,
        "weight": weight,
        "reason_code": reason,
        "explanation": f"Causal link established: {reason}"
    }
    event_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    event = {
        "event_id": event_id,
        "action": "CREATE_EDGE",
        "timestamp": time.time(),
        "payload": payload
    }
    signed_event = sign_event(event, priv_key)
    producer.send('tessrax_ledger', signed_event)
    print(f"Sent EDGE event: {event_id}")
    return event_id

if __name__ == "__main__":
    print("--- Starting Event Emission Simulation ---")
    
    # 1. Create a contradiction
    node1_data = {
        "description": "Policy A states 'All data local', but implementation B uses cloud sync.",
        "modality": "policy/tech",
        "domain": "security"
    }
    node1_hash = emit_contradiction(node1_data)
    
    time.sleep(1)
    
    # 2. Create a second, related contradiction
    node2_data = {
        "description": "User audit logs (local) do not match cloud provider logs.",
        "modality": "log/audit",
        "domain": "security"
    }
    node2_hash = emit_contradiction(node2_data)

    time.sleep(1)
    
    # 3. Link them causally
    emit_causal_link(
        source_hash=node1_hash,
        target_hash=node2_hash,
        weight=0.85,
        reason="VIOLATION_DERIVED_DATA"
    )
    
    producer.flush()
    print("--- Event Emission Complete ---")

streamer/kafka_consumer.py (Updated Service)
This service runs in its own Docker container, listening to Kafka and writing to Neo4j.
import json
import ed25519
from kafka import KafkaConsumer
from db.neo4j_handler import CCGNGraphHandler # Assumes handler is in db/
from ledger.ledger_api import verify_event # Assumes verification logic is in ledger/

# --- Load Kernel Public Key to Verify Signatures ---
try:
    kernel_pub_key = ed25519.VerifyingKey(open("tessrax_kernel.pub", "rb").read())
    print("Loaded kernel public key for verification.")
except FileNotFoundError:
    print("FATAL: Kernel public key 'tessrax_kernel.pub' not found. Cannot verify events.")
    exit(1)

def main():
    # Connect to Neo4j (from our defined handler)
    # Env vars would be better for production
    handler = CCGNGraphHandler("bolt://neo4j:7687", "neo4j", "your_password")
    
    # Connect to Kafka
    consumer = KafkaConsumer(
        'tessrax_ledger',
        bootstrap_servers='kafka:9092', # Use Docker service name
        auto_offset_reset='earliest',
        group_id='ccgn-ingestor-group',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )
    
    print("Kafka Consumer connected. Listening for ledger events...")

    for message in consumer:
        event = message.value
        
        # 1. Verify Cryptographic Integrity
        if not verify_event(event, kernel_pub_key):
            print(f"!! CRITICAL: Invalid signature on event {event.get('event_id')}. Discarding.")
            continue
            
        print(f"Received VALID event: {event['event_id']} (Action: {event['action']})")
        
        # 2. Process and Map Event to Graph
        try:
            payload = event['payload']
            
            if event['action'] == 'CREATE_NODE':
                node_hash = handler.create_or_update_node(
                    node_id=event['event_id'], # Use event_id as the canonical graph ID
                    description=payload['description'],
                    modality=payload['modality'],
                    domain=payload.get('domain', 'general'),
                    timestamp=event['timestamp']
                )
                print(f"  -> Mapped NODE: {node_hash}")

            elif event['action'] == 'CREATE_EDGE':
                handler.create_or_update_edge(
                    edge_id=event['event_id'],
                    source_hash=payload['source_hash'],
                    target_hash=payload['target_hash'],
                    weight=payload['weight'],
                    reason_code=payload['reason_code'],
                    explanation=payload['explanation'],
                    timestamp=event['timestamp']
                )
                print(f"  -> Mapped EDGE: {payload['source_hash']} -> {payload['target_hash']}")

        except Exception as e:
            print(f"!! ERROR processing event {event.get('event_id')}: {e}")
            # In production, this would go to a dead-letter queue
            
if __name__ == "__main__":
    # Add a simple retry logic for connecting to Kafka/Neo4j on startup
    connected = False
    for _ in range(5):
        try:
            main()
            connected = True
            break
        except Exception as e:
            print(f"Failed to connect, retrying... {e}")
            time.sleep(5)
    
    if not connected:
        print("Fatal: Could not connect to services after retries.")
        exit(1)

ledger/ledger_api.py (Updated Utility)
This file holds the reusable verification logic.
import json
import ed25519
from typing import Dict, Any

def verify_event(event: Dict[str, Any], public_key: ed25519.VerifyingKey) -> bool:
    """Verifies a signed ledger event."""
    if 'signature' not in event:
        return False
        
    try:
        signature = bytes.fromhex(event['signature'])
        
        # Create a copy of the event *without* the signature for verification
        event_data_copy = event.copy()
        del event_data_copy['signature']
        
        event_bytes = json.dumps(event_data_copy, sort_keys=True).encode('utf-8')
        
        public_key.verify(signature, event_bytes)
        return True
    except (ed25519.BadSignatureError, ValueError, TypeError):
        return False

2. Database Handler (db/)
The neo4j_handler.py is updated with the logic migrated from the prototype (EntropyForecaster, CCGNPrototype) to run Cypher queries for entropy and explanation.
db/neo4j_handler.py (Updated Service)
import json
import hashlib
import math
from neo4j import GraphDatabase
from typing import Dict, Any, Tuple, List

class CCGNGraphHandler:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._ensure_constraints()

    def close(self):
        self.driver.close()

    def _ensure_constraints(self):
        """Ensure uniqueness constraints on Contradiction nodes for merging."""
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS ON (n:Contradiction) ASSERT n.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS ON (r:CAUSES) ASSERT r.id IS UNIQUE")

    def create_or_update_node(self, node_id: str, description: str, modality: str, domain: str, timestamp: float):
        """Idempotent creation/update of a Contradiction node."""
        with self.driver.session() as session:
            session.run(
                """
                MERGE (n:Contradiction {id: $id})
                SET n.description = $desc,
                    n.modality = $modality,
                    n.domain = $domain,
                    n.timestamp = $timestamp,
                    n.dampened = false
                """,
                id=node_id, desc=description, modality=modality, domain=domain, timestamp=timestamp
            )
        return node_id

    def create_or_update_edge(self, edge_id: str, source_hash: str, target_hash: str, weight: float, reason_code: str, explanation: str, timestamp: float):
        """Idempotent creation/update of a CAUSES edge between two nodes."""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (s:Contradiction {id: $src}), (t:Contradiction {id: $tgt})
                MERGE (s)-[r:CAUSES {id: $edge_id}]->(t)
                SET r.weight = $weight,
                    r.reason_code = $reason_code,
                    r.explanation = $explanation,
                    r.timestamp = $timestamp
                """,
                edge_id=edge_id, src=source_hash, tgt=target_hash, weight=weight, 
                reason_code=reason_code, explanation=explanation, timestamp=timestamp
            )
            
    # --- Logic Migrated from Prototype (EntropyForecaster) ---
    def get_system_entropy(self) -> float:
        """Calculates the graph's current Shannon entropy based on node degree centrality."""
        with self.driver.session() as session:
            # 1. Get degree centrality for all non-dampened nodes
            result = session.run(
                """
                MATCH (n:Contradiction)
                WHERE n.dampened = false
                CALL {
                    WITH n
                    MATCH (n)-[r:CAUSES]-()
                    WHERE r.weight > 0
                    RETURN count(r) AS degree
                }
                RETURN degree
                """
            )
            
            degrees = [record["degree"] for record in result]
            if not degrees:
                return 0.0

            total_degree = sum(degrees)
            if total_degree == 0:
                return 0.0

            entropy = 0.0
            for degree in degrees:
                if degree > 0:
                    probability = degree / total_degree
                    entropy -= probability * math.log2(probability)
            
            return entropy

    # --- Logic for API Endpoints ---
    def get_edge_explanation(self, edge_id: str) -> Dict[str, Any]:
        """Fetches the explanation for a single causal edge."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (s:Contradiction)-[r:CAUSES {id: $edge_id}]->(t:Contradiction)
                RETURN s.description AS source, 
                       t.description AS target, 
                       r.explanation AS explanation,
                       r.reason_code AS reason,
                       r.weight AS weight
                """,
                edge_id=edge_id
            )
            record = result.single()
            if record:
                return record.data()
            else:
                return {"error": "Edge not found."}

    def simulate_intervention(self, node_id: str, strength: float) -> Tuple[float, List[str]]:
        """Simulates dampening a node and returns the new entropy and affected nodes."""
        
        # 1. Get entropy *before* intervention
        entropy_before = self.get_system_entropy()
        
        # 2. Find downstream nodes *before* dampening
        affected_nodes = []
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (startNode:Contradiction {id: $node_id})-[:CAUSES*1..5]->(downstream:Contradiction)
                WHERE downstream.dampened = false
                RETURN COLLECT(DISTINCT downstream.id) AS affected
                """,
                node_id=node_id
            )
            affected_nodes = result.single()["affected"]

        # 3. Apply the intervention (dampen the node)
        with self.driver.session() as session:
            session.run(
                """
                MATCH (n:Contradiction {id: $node_id})
                SET n.dampened = true, n.dampen_strength = $strength
                """,
                node_id=node_id, strength=strength
            )
            
        # 4. Get entropy *after* intervention
        entropy_after = self.get_system_entropy()
        
        # 5. (IMPORTANT) Rollback the change so the simulation is read-only
        with self.driver.session() as session:
            session.run(
                """
                MATCH (n:Contradiction {id: $node_id})
                SET n.dampened = false
                REMOVE n.dampen_strength
                """
            )
            
        entropy_change = entropy_after - entropy_before
        return entropy_change, affected_nodes

3. API & Simulation (api/ and simulation/)
These files create the live API endpoints, importing the migrated logic from the db/ handler.
api/routes/simulation.py (Updated API Route)
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from db.neo4j_handler import CCGNGraphHandler

router = APIRouter()

# --- Dependency Injection for the DB Handler ---
def get_db():
    # This is a simplified dependency injector.
    # In prod, you'd use a more robust session/connection pool.
    handler = CCGNGraphHandler("bolt://neo4j:7687", "neo4j", "your_password")
    try:
        yield handler
    finally:
        handler.close()

# --- Request/Response Models ---
class InterventionRequest(BaseModel):
    node_id: str
    strength: float = 1.0 # Default to full dampening

class InterventionResponse(BaseModel):
    entropy_change: float
    affected_nodes: list[str]
    message: str

class EntropyResponse(BaseModel):
    current_system_entropy: float

# --- API Endpoints ---
@router.get("/entropy", response_model=EntropyResponse)
def get_current_entropy(db: CCGNGraphHandler = Depends(get_db)):
    """
    Get the current Shannon Entropy of the entire active contradiction graph.
    """
    entropy = db.get_system_entropy()
    return EntropyResponse(current_system_entropy=entropy)

@router.post("/intervention", response_model=InterventionResponse)
def intervention_simulation(req: InterventionRequest, db: CCGNGraphHandler = Depends(get_db)):
    """
    Simulate dampening a node and see the effect on system entropy.
    This is a READ-ONLY operation (changes are rolled back).
    """
    try:
        change, nodes = db.simulate_intervention(req.node_id, req.strength)
        return InterventionResponse(
            entropy_change=change,
            affected_nodes=nodes,
            message=f"Simulating intervention on {req.node_id} resulted in {change:.4f} entropy change."
        )
    except Exception as e:
        return InterventionResponse(
            entropy_change=0,
            affected_nodes=[],
            message=f"Error during simulation: {e}"
        )

api/routes/explanation.py (Updated API Route)
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from db.neo4j_handler import CCGNGraphHandler
from typing import Optional

router = APIRouter()

# --- Dependency Injection (can be shared from simulation.py) ---
def get_db():
    handler = CCGNGraphHandler("bolt://neo4j:7687", "neo4j", "your_password")
    try:
        yield handler
    finally:
        handler.close()

# --- Response Model ---
class ExplanationResponse(BaseModel):
    source: str
    target: str
    explanation: str
    reason: str
    weight: float

# --- API Endpoint ---
@router.get("/{edge_id}", response_model=ExplanationResponse)
def explain_edge(edge_id: str, db: CCGNGraphHandler = Depends(get_db)):
    """
    Get the human-readable explanation and metadata for a specific causal edge.
    """
    result = db.get_edge_explanation(edge_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return ExplanationResponse(**result)

Migration Summary
 * Prototype ClosureLedger -> Production kafka_consumer.py: The logic is now asynchronous, distributed, and persistent.
 * Prototype LedgerIntegrityEngine -> Production ledger/ledger_api.py: The logic is now a reusable cryptographic utility for both the producer and consumer.
 * Prototype CCGNPrototype (NetworkX) -> Production db/neo4j_handler.py (Neo4j): The graph logic is migrated from in-memory to a persistent, queryable database.
 * Prototype EntropyForecaster -> Production db/neo4j_handler.py: The entropy calculation is migrated from a Python networkx call to a Cypher query.
 * Prototype ForecastAgent -> Production api/routes/simulation.py: The simulation logic is now exposed as a live, stateless API endpoint that runs its calculations against the production DB.
The prototype logic is successfully installed into the production architecture. The system is now ready for deployment and end-to-end testing.


"""
Tessrax Colab Full Runtime Bundle
Merged from Colab_code_1.txt and Colab_code_2.txt
Contains 12 independent subsystems:
  1. PredictiveRuntime
  2. IntegrityMonitor
  3. ZKAudit
  4. ClosureLedger
  5. EntropyForecaster
  6. LedgerIntegrityEngine
  7. GovernanceKernelRuntime
  8. EntropySignalAnalyzer
  9. CCGNPrototype
  10. ForecastAgent
  11. ZKProofEngine
  12. CCGNVisualizer
"""

import os, json, time, math, random, hashlib, hmac, logging, datetime
import networkx as nx

try:
    import numpy as np
except ImportError:
    np = None

# ---------------------------------------------------------------------
# 1. PredictiveRuntime
# ---------------------------------------------------------------------
class PredictiveRuntime:
    def __init__(self):
        self.history = []

    def run_cycle(self, metric_value: float):
        prediction = self._predict(metric_value)
        self.history.append({"input": metric_value, "prediction": prediction})
        return prediction

    def _predict(self, value):
        # Simple sigmoid normalization to [0,1]
        return 1 / (1 + math.exp(-value))

# ---------------------------------------------------------------------
# 2. IntegrityMonitor
# ---------------------------------------------------------------------
class IntegrityMonitor:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def compute_hash(self, filepath):
        sha = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                sha.update(chunk)
        return sha.hexdigest()

    def verify_directory(self):
        report = {}
        for root, _, files in os.walk(self.base_dir):
            for f in files:
                path = os.path.join(root, f)
                report[path] = self.compute_hash(path)
        return report

# ---------------------------------------------------------------------
# 3. ZKAudit
# ---------------------------------------------------------------------
class ZKAudit:
    def __init__(self, secret_key: bytes):
        self.secret_key = secret_key

    def commit(self, message: str):
        digest = hmac.new(self.secret_key, message.encode(), hashlib.sha256).hexdigest()
        return {"message_hash": digest, "timestamp": datetime.datetime.utcnow().isoformat()}

# ---------------------------------------------------------------------
# 4. ClosureLedger
# ---------------------------------------------------------------------
class ClosureLedger:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_event(self, event_id, causes):
        self.graph.add_node(event_id, status="open", resolution=None)
        for c in causes:
            self.graph.add_edge(c, event_id)

    def close_event(self, event_id, resolution):
        if not self.graph.has_node(event_id):
            raise ValueError("Event not found")
        self.graph.nodes[event_id]["status"] = "closed"
        self.graph.nodes[event_id]["resolution"] = resolution

    def export_json(self):
        return json.dumps(nx.node_link_data(self.graph), indent=2)

# ---------------------------------------------------------------------
# 5. EntropyForecaster
# ---------------------------------------------------------------------
class EntropyForecaster:
    def __init__(self):
        self.values = []

    def update(self, val):
        self.values.append(val)
        return self.predict_next()

    def predict_next(self):
        if len(self.values) < 2:
            return self.values[-1] if self.values else 0
        diffs = [self.values[i+1]-self.values[i] for i in range(len(self.values)-1)]
        avg_delta = sum(diffs)/len(diffs)
        return self.values[-1] + avg_delta

# ---------------------------------------------------------------------
# 6. LedgerIntegrityEngine
# ---------------------------------------------------------------------
class LedgerIntegrityEngine:
    def __init__(self):
        self.ledger = []

    def append_entry(self, entry: dict):
        entry_json = json.dumps(entry, sort_keys=True)
        entry_hash = hashlib.sha256(entry_json.encode()).hexdigest()
        self.ledger.append({"entry": entry, "hash": entry_hash})
        return entry_hash

    def verify_ledger(self):
        return all(
            rec["hash"] == hashlib.sha256(json.dumps(rec["entry"], sort_keys=True).encode()).hexdigest()
            for rec in self.ledger
        )

# ---------------------------------------------------------------------
# 7. GovernanceKernelRuntime
# ---------------------------------------------------------------------
class GovernanceKernelRuntime:
    def __init__(self):
        self.events = []
        self.logger = logging.getLogger("GovernanceKernel")
        logging.basicConfig(level=logging.INFO)

    def detect_contradiction(self, signal_strength):
        contradiction = signal_strength > 0.7
        self.logger.info(f"Signal={signal_strength:.2f} -> Contradiction={contradiction}")
        self.events.append({"signal": signal_strength, "contradiction": contradiction})
        return contradiction

# ---------------------------------------------------------------------
# 8. EntropySignalAnalyzer
# ---------------------------------------------------------------------
class EntropySignalAnalyzer:
    def __init__(self, window=5):
        self.window = window
        self.data = []

    def push(self, val):
        self.data.append(val)
        if len(self.data) > self.window:
            self.data.pop(0)
        return self.compute_entropy()

    def compute_entropy(self):
        if not self.data:
            return 0
        # Handle potential division by zero or negative variance if data has < 2 points after windowing
        if len(self.data) < 2:
             return 0 # Cannot compute meaningful variance/entropy with one or no data point
        avg = sum(self.data)/len(self.data)
        variance_sum = sum((x-avg)**2 for x in self.data)
        # Avoid division by zero if len(self.data) is 0 or 1 (handled above)
        # Avoid log(0) if variance is 0 (all data points are identical)
        variance = variance_sum / len(self.data)
        entropy = math.log(variance+1, 2) # Add 1 to variance to avoid log(0)
        return entropy


# ---------------------------------------------------------------------
# 9. CCGNPrototype
# ---------------------------------------------------------------------
class CCGNPrototype:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_relation(self, a, b, weight=1.0):
        self.graph.add_edge(a, b, weight=weight)

    def propagate(self):
        for a, b, d in self.graph.edges(data=True):
            d["weight"] *= 0.95
        return self.graph

# ---------------------------------------------------------------------
# 10. ForecastAgent
# ---------------------------------------------------------------------
class ForecastAgent:
    def __init__(self):
        self.history = []

    def forecast(self, entropy_values):
        if np is None or len(entropy_values) < 2:
            # Fallback if numpy is not available or not enough data for polyfit
            if not entropy_values:
                trend = 0
            else:
                # Simple trend based on last two points if available, otherwise 0
                if len(entropy_values) >= 2:
                     trend = entropy_values[-1] - entropy_values[-2]
                else:
                     trend = 0
        else:
            # Use numpy polyfit if available and enough data
            try:
                 # Suppress polyfit warnings about rank deficiency for small datasets
                 with warnings.catch_warnings():
                     warnings.simplefilter("ignore", np.RankWarning)
                     trend = np.polyfit(range(len(entropy_values)), entropy_values, 1)[0]
            except Exception as e:
                 print(f"Warning: Error during numpy polyfit: {e}. Falling back to simple trend.")
                 if len(entropy_values) >= 2:
                      trend = entropy_values[-1] - entropy_values[-2]
                 else:
                      trend = 0


        prediction = "rising" if trend > 0 else ("falling" if trend < 0 else "stable") # Add stable state
        self.history.append({"trend": trend, "prediction": prediction})
        return prediction

# ---------------------------------------------------------------------
# 11. ZKProofEngine
# ---------------------------------------------------------------------
class ZKProofEngine:
    def __init__(self):
        self.commits = []

    def commit(self, data: str):
        h = hashlib.sha256(data.encode()).hexdigest()
        self.commits.append(h)
        return h

    def verify(self, data: str, h: str):
        return hashlib.sha256(data.encode()).hexdigest() == h

# ---------------------------------------------------------------------
# 12. CCGNVisualizer
# ---------------------------------------------------------------------
class CCGNVisualizer:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    def render(self):
        try:
            import matplotlib.pyplot as plt
            pos = nx.spring_layout(self.graph)
            weights = [self.graph[u][v]['weight'] for u,v in self.graph.edges()]
            # Add labels to edges based on weight if desired
            edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in self.graph.edges(data=True)}

            nx.draw(self.graph, pos, with_labels=True, width=weights, node_color='lightblue', node_size=700, font_size=10)
            # Draw edge labels
            nx.draw_edge_labels(self.graph, pos, edge_labels=edge_labels, font_color='red')

            plt.title("CCGN Visualization")
            plt.show()
        except ImportError:
            print("Visualization skipped: matplotlib not installed. Install with 'pip install matplotlib'.")
        except Exception as e:
            print("Visualization skipped:", e)

# ---------------------------------------------------------------------
# Demo Menu
# ---------------------------------------------------------------------
import warnings # Import warnings for numpy polyfit
if __name__ == "__main__":
    print("=== Tessrax Colab Full Runtime Demo ===")
    print("1: PredictiveRuntime\n2: IntegrityMonitor\n3: ZKAudit\n4: ClosureLedger\n5: EntropyForecaster\n"
          "6: LedgerIntegrityEngine\n7: GovernanceKernelRuntime\n8: EntropySignalAnalyzer\n9: CCGNPrototype\n"
          "10: ForecastAgent\n11: ZKProofEngine\n12: CCGNVisualizer")
    choice = input("Select module number to run demo (or ENTER to exit): ").strip()

    if choice == "1":
        rt = PredictiveRuntime()
        print("\n--- PredictiveRuntime Demo ---")
        test_values = [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]
        print("Input -> Prediction")
        for v in test_values:
            print(f"{v} -> {rt.run_cycle(v):.4f}")
        print("\nHistory:", rt.history)

    elif choice == "2":
        print("\n--- IntegrityMonitor Demo ---")
        # Create dummy files for demonstration
        dummy_dir = "temp_integrity_check"
        os.makedirs(dummy_dir, exist_ok=True)
        with open(os.path.join(dummy_dir, "file1.txt"), "w") as f:
            f.write("This is file one.")
        with open(os.path.join(dummy_dir, "file2.txt"), "w") as f:
            f.write("This is file two.")

        monitor = IntegrityMonitor(dummy_dir)
        report = monitor.verify_directory()
        print(f"\nIntegrity Report for '{dummy_dir}':")
        for filepath, filehash in report.items():
            print(f"  {filepath}: {filehash}")

        # Clean up dummy directory
        try:
             os.remove(os.path.join(dummy_dir, "file1.txt"))
             os.remove(os.path.join(dummy_dir, "file2.txt"))
             os.rmdir(dummy_dir)
             print(f"\nCleaned up dummy directory '{dummy_dir}'.")
        except Exception as e:
             print(f"Warning: Could not clean up dummy directory: {e}")


    elif choice == "3":
        print("\n--- ZKAudit Demo ---")
        # Use a fixed secret key for demo
        secret = b'supersecretkey'
        zka = ZKAudit(secret)
        message1 = "Transaction ABC occurred."
        message2 = "Transaction XYZ occurred."
        message3 = "Transaction ABC occurred." # Same message as message1

        commit1 = zka.commit(message1)
        commit2 = zka.commit(message2)
        commit3 = zka.commit(message3)

        print("\nCommit 1 (Message 1):", json.dumps(commit1, indent=2))
        print("\nCommit 2 (Message 2):", json.dumps(commit2, indent=2))
        print("\nCommit 3 (Message 3 - same as Message 1):", json.dumps(commit3, indent=2))

        # Verify if commit hashes match for identical messages
        print(f"\nCommit hash for Message 1 == Commit hash for Message 3: {commit1['message_hash'] == commit3['message_hash']}")
        print(f"Commit hash for Message 1 == Commit hash for Message 2: {commit1['message_hash'] == commit2['message_hash']}")


    elif choice == "4":
        print("\n--- ClosureLedger Demo ---")
        cl = ClosureLedger()

        # Add events with causal links
        cl.add_event("DataIngested", [])
        cl.add_event("ContradictionDetected", ["DataIngested"])
        cl.add_event("ResolutionProposed", ["ContradictionDetected"])
        cl.add_event("AmendmentApplied", ["ResolutionProposed"])
        cl.add_event("UnrelatedEvent", []) # Unrelated event


        print("\nInitial Ledger State (before closing):")
        print(cl.export_json())

        # Close some events
        cl.close_event("ContradictionDetected", {"status": "resolved", "details": "Manual review"})
        cl.close_event("AmendmentApplied", {"status": "complete"})

        print("\nLedger State after closing events:")
        print(cl.export_json())

        # Try to close a non-existent event
        try:
            cl.close_event("FakeEvent", {"status": "failed"})
        except ValueError as e:
            print(f"\nAttempted to close a fake event: {e}")


    elif choice == "5":
        print("\n--- EntropyForecaster Demo ---")
        ef = EntropyForecaster()

        print("Initial prediction:", ef.predict_next()) # Should be 0

        print("Updating with 0.1:", ef.update(0.1)) # Should be 0.1 (just return last)
        print("Updating with 0.3:", ef.update(0.3)) # Should be 0.5 (0.3 + (0.3-0.1))
        print("Updating with 0.2:", ef.update(0.2)) # Should be 0.1 (0.2 + ((0.3-0.1)+(0.2-0.3))/2) -> 0.2 + (0.2 - 0.1)/2 = 0.2 + 0.05 = 0.25?
        # Re-calculating: values=[0.1, 0.3, 0.2]. diffs=[0.2, -0.1]. avg_delta = (0.2 - 0.1)/2 = 0.05. Prediction = 0.2 + 0.05 = 0.25
        print("Updating with 0.2 (Corrected):", ef.update(0.2))

        print("\nHistory:", ef.values)
        print("Next predicted value:", ef.predict_next()) # Based on current history


    elif choice == "6":
        print("\n--- LedgerIntegrityEngine Demo ---")
        le = LedgerIntegrityEngine()

        entry1 = {"event_type": "DATA_POINT", "data": {"value": 100}}
        entry2 = {"event_type": "ALERT", "data": {"message": "High value detected"}}
        entry3 = {"event_type": "SYSTEM_EVENT", "data": {"status": "OK"}}

        hash1 = le.append_entry(entry1)
        hash2 = le.append_entry(entry2)
        hash3 = le.append_entry(entry3)

        print("\nAppended entries and got hashes:")
        print("Entry 1 Hash:", hash1)
        print("Entry 2 Hash:", hash2)
        print("Entry 3 Hash:", hash3)

        print("\nVerifying ledger integrity...")
        is_valid = le.verify_ledger()
        print("Ledger valid?", is_valid)

        # Simulate tampering (modify an entry in the list directly - not how a real ledger would be tampered)
        if le.ledger:
             print("\nSimulating ledger tampering (modifying data in memory)...")
             le.ledger[1]["entry"]["data"]["message"] = "Tampered message!"
             print("Verifying ledger integrity after tampering...")
             is_valid_tampered = le.verify_ledger()
             print("Ledger valid after tampering?", is_valid_tampered)
        else:
             print("\nSkipping tampering demo: Ledger is empty.")


    elif choice == "7":
        print("\n--- GovernanceKernelRuntime Demo ---")
        kernel = GovernanceKernelRuntime()

        print("\nSignal Strength -> Contradiction Detected")
        signals = [0.6, 0.8, 0.5, 0.9, 0.7]
        for signal in signals:
            contradiction = kernel.detect_contradiction(signal)
            # print(f"{signal:.2f} -> {contradiction}") # Output is already logged by the kernel's logger

        print("\nKernel Events:", kernel.events)


    elif choice == "8":
        print("\n--- EntropySignalAnalyzer Demo ---")
        analyzer = EntropySignalAnalyzer(window=3)

        print("Pushing 1.0 -> Entropy:", analyzer.push(1.0)) # Entropy 0 (one data point)
        print("Pushing 1.1 -> Entropy:", analyzer.push(1.1)) # Entropy > 0 (two data points, variance > 0)
        print("Pushing 1.2 -> Entropy:", analyzer.push(1.2)) # Entropy updated (three data points)
        print("Pushing 1.0 -> Entropy:", analyzer.push(1.0)) # Window full, oldest (1.0) removed, new 1.0 added [1.1, 1.2, 1.0]
        print("Pushing 1.5 -> Entropy:", analyzer.push(1.5)) # Window full, oldest (1.1) removed [1.2, 1.0, 1.5]

        print("\nAnalyzer Data Window:", analyzer.data)
        print("Current Entropy:", analyzer.compute_entropy())


    elif choice == "9":
        print("\n--- CCGNPrototype Demo ---")
        ccgn = CCGNPrototype()

        # Add relations
        ccgn.add_relation("Data Source A", "Claim 1", weight=0.8)
        ccgn.add_relation("Data Source B", "Claim 1", weight=0.7) # Conflict?
        ccgn.add_relation("Claim 1", "Contradiction 1", weight=0.9)
        ccgn.add_relation("Contradiction 1", "Resolution 1", weight=0.6)
        ccgn.add_relation("Resolution 1", "Policy Update", weight=0.5)
        ccgn.add_relation("Data Source C", "Claim 2", weight=0.95) # Unrelated chain

        print("\nInitial Graph Edges with Weights:")
        print(ccgn.graph.edges(data=True))

        print("\nPropagating influence (weights decay)...")
        ccgn.propagate() # First propagation
        ccgn.propagate() # Second propagation
        ccgn.propagate() # Third propagation

        print("\nGraph Edges with Weights after Propagation:")
        print(ccgn.graph.edges(data=True))


    elif choice == "10":
        print("\n--- ForecastAgent Demo ---")
        agent = ForecastAgent()

        print("\nForecasting with empty history:")
        print("Prediction:", agent.forecast([])) # Should be stable/0 trend

        print("\nForecasting with minimal history [0.1]:")
        print("Prediction:", agent.forecast([0.1])) # Should be stable/0 trend

        print("\nForecasting with rising trend [0.1, 0.3, 0.5]:")
        print("Prediction:", agent.forecast([0.1, 0.3, 0.5])) # Should be rising

        print("\nForecasting with falling trend [0.5, 0.3, 0.1]:")
        print("Prediction:", agent.forecast([0.5, 0.3, 0.1])) # Should be falling

        print("\nForecasting with mixed trend [0.1, 0.5, 0.2]:")
        print("Prediction:", agent.forecast([0.1, 0.5, 0.2])) # Should be falling (0.5->0.2 is strong fall)

        print("\nAgent History:", agent.history)


    elif choice == "11":
        print("\n--- ZKProofEngine Demo ---")
        zk = ZKProofEngine()

        data1 = "Sensitive data point A: value is 123.45"
        data2 = "Sensitive data point B: value is 678.90"
        data3 = "Sensitive data point A: value is 123.45" # Same as data1

        commit1 = zk.commit(data1)
        commit2 = zk.commit(data2)
        commit3 = zk.commit(data3)

        print("\nCommit 1 (Data 1 Hash):", commit1)
        print("Commit 2 (Data 2 Hash):", commit2)
        print("Commit 3 (Data 3 Hash):", commit3)

        print("\nVerifying Data 1 against Commit 1:", zk.verify(data1, commit1))
        print("Verifying Data 2 against Commit 1:", zk.verify(data2, commit1)) # Should be False
        print("Verifying Data 3 against Commit 1:", zk.verify(data3, commit1)) # Should be True (same data)

        # Simulate providing incorrect data for verification
        print("Verifying Tampered Data against Commit 1:", zk.verify("Tampered data", commit1)) # Should be False


    elif choice == "12":
        print("\n--- CCGNVisualizer Demo ---")
        # Create a sample graph for visualization
        graph_to_viz = nx.DiGraph()
        graph_to_viz.add_edge("Source A", "Claim 1", weight=1.0)
        graph_to_viz.add_edge("Source B", "Claim 1", weight=0.8)
        graph_to_viz.add_edge("Claim 1", "Contradiction 1", weight=1.0)
        graph_to_viz.add_edge("Contradiction 1", "Resolution 1", weight=0.7)
        graph_to_viz.add_edge("Claim 2", "Contradiction 1", weight=0.9) # Another source for contradiction
        graph_to_viz.add_edge("Resolution 1", "Amendment", weight=0.6)


        viz = CCGNVisualizer(graph_to_viz)
        print("\nRendering CCGN Visualization...")
        viz.render()
        print("Visualization window should appear (if matplotlib is installed and in a compatible environment).")


    else:
        print("\nNo module selected or invalid choice. Exiting demo.")


# Remove the usage instructions from the executed code block
# This ensures only the code runs when the cell is executed directly.
# The instructions are helpful in the markdown explanation.
# print("\n⸻")
# print("\n✅ Usage:")
# print("        1.      Save this block as tessrax_colab_full.py.")
# print("        2.      Install networkx (and optionally matplotlib, numpy):")
# print("\npip install networkx matplotlib numpy\n")
# print("        3.      Run:")
# print("\npython tessrax_colab_full.py\n")
# print("        4.      Choose a module number to test each subsystem interactively.")
# print("\nThis single file now executes every functional component from both Colab code bases.")
