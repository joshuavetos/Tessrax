"""
Tessrax Orchestrator v1.0
Central coordination layer that decides which engines to run and when.
This is the "nervous system" that makes all other components work as one organism.

Author: Joshua Vetos / Tessrax LLC  
License: CC BY 4.0
"""

import json
from datetime import datetime
from pathlib import Path
from governance_kernel import GovernanceKernel
from agent_governance_demo import TessraxGovernanceAgent, SimpleSemanticEngine
from scaffolding_engine import record_interaction
from corporate_frienthropy import calculate_cfi  # Assuming this function exists

class TessraxOrchestrator:
    """
    The conductor. Observes events, decides responses, coordinates subsystems.
    """
    
    def __init__(self):
        print("🧠 Initializing Tessrax Orchestrator...")
        self.kernel = GovernanceKernel()
        self.semantic_engine = SimpleSemanticEngine()
        self.agent = TessraxGovernanceAgent("Orchestrator-Agent", self.semantic_engine, self.kernel)
        
        # Thresholds and rules
        self.cfi_threshold = 0.4
        self.violation_alert_threshold = 10
        
        print("✅ Orchestrator ready\n")
    
    # ============================================================
    # Main Processing Loop
    # ============================================================
    
    def process(self, event):
        """
        Universal event handler. Routes to appropriate subsystems.
        """
        print(f"📥 Processing: {event.get('type', 'UNKNOWN')}")
        
        # Always log to governance ledger
        self.kernel.append_event(event)
        
        # Route based on event type
        event_type = event.get('type') or event.get('event')
        
        if 'DESIGN_DECISION' in event_type:
            self._handle_design_decision(event)
            
        elif 'CLAIM' in event_type or 'REPORT' in event_type:
            self._handle_semantic_content(event)
            
        elif 'CFI' in event_type or 'FRIENTHROPY' in event_type:
            self._handle_cfi_report(event)
            
        elif 'VIOLATION' in event_type:
            self._handle_violation(event)
        
        # Always check system health after processing
        self._self_diagnose()
    
    # ============================================================
    # Event Handlers
    # ============================================================
    
    def _handle_design_decision(self, event):
        """Design decisions get policy checked and logged."""
        print("  → Design decision detected")
        # Policy check happens automatically in governance_kernel
        # Just ensure it's in the scaffolding log
        try:
            record_interaction(
                prompt=event.get('prompt', 'Auto-logged'),
                response=event.get('response', 'System event'),
                tags=event.get('tags', []),
                file_changed=event.get('file_changed')
            )
        except Exception as e:
            print(f"  ⚠️ Scaffolding log failed: {e}")
    
    def _handle_semantic_content(self, event):
        """Claims and reports get analyzed for contradictions."""
        print("  → Semantic content detected, analyzing...")
        self.agent.process_event(event)
        # Agent automatically logs AGENT_ANALYSIS_REPORT to kernel
    
    def _handle_cfi_report(self, event):
        """CFI reports trigger governance if threshold exceeded."""
        cfi_score = event.get('cfi_score', 0)
        print(f"  → CFI score: {cfi_score:.2f}")
        
        if cfi_score > self.cfi_threshold:
            print(f"  🚨 CFI exceeds threshold {self.cfi_threshold}")
            violation = {
                'event': 'CORPORATE_PROMISE_VIOLATION',
                'cfi_score': cfi_score,
                'threshold': self.cfi_threshold,
                'enforcement': 'quorum'
            }
            self.kernel.append_event(violation)
    
    def _handle_violation(self, event):
        """Violations are logged and monitored for patterns."""
        print("  ⚠️ Violation logged")
        # Governance kernel handles escalation automatically
    
    # ============================================================
    # Self-Diagnosis
    # ============================================================
    
    def _self_diagnose(self):
        """
        System examines its own recent behavior.
        Alerts if patterns suggest problems.
        """
        recent = self.kernel.ledger[-100:] if len(self.kernel.ledger) > 100 else self.kernel.ledger
        
        # Count violations
        violations = [e for e in recent if 'VIOLATION' in e.get('event', '')]
        
        if len(violations) >= self.violation_alert_threshold:
            alert = {
                'event': 'SYSTEM_HEALTH_ALERT',
                'reason': 'High violation rate detected',
                'violation_count': len(violations),
                'window_size': len(recent),
                'timestamp': datetime.utcnow().isoformat()
            }
            self.kernel.append_event(alert)
            print(f"  🚨 SYSTEM ALERT: {len(violations)} violations in last {len(recent)} events")
    
    # ============================================================
    # Batch Processing
    # ============================================================
    
    def process_stream(self, events):
        """Process multiple events in sequence."""
        print(f"\n🔄 Processing stream of {len(events)} events...\n")
        for event in events:
            self.process(event)
            print()  # Blank line between events
        print("✅ Stream processing complete\n")
    
    # ============================================================
    # Status Report
    # ============================================================
    
    def status(self):
        """Return current system state."""
        return {
            'total_events': len(self.kernel.ledger),
            'recent_violations': len([e for e in self.kernel.ledger[-100:] 
                                     if 'VIOLATION' in e.get('event', '')]),
            'agent_analyses': len(self.agent.reports),
            'ledger_integrity': self.kernel.ledger[-1]['hash'] if self.kernel.ledger else None
        }


# ============================================================
# Demo / Test Harness
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("TESSRAX ORCHESTRATOR DEMO")
    print("="*60 + "\n")
    
    orchestrator = TessraxOrchestrator()
    
    # Simulated event stream
    demo_events = [
        {
            'type': 'DESIGN_DECISION_RECORDED',
            'file_changed': 'tessrax_orchestrator.py',
            'tags': ['orchestration', 'meta'],
            'prompt': 'Build central coordinator',
            'response': 'Created TessraxOrchestrator class'
        },
        {
            'type': 'CLAIM',
            'id': 'claim-001',
            'payload': {'text': 'This statement is false.'}
        },
        {
            'type': 'CFI_REPORT',
            'cfi_score': 0.52,
            'period': '2025-Q4'
        },
        {
            'type': 'DESIGN_DECISION_RECORDED',
            'file_changed': 'fork_engine.py',
            'tags': ['governance'],  # Missing 'fork' tag - will violate policy
        }
    ]
    
    orchestrator.process_stream(demo_events)
    
    print("="*60)
    print("SYSTEM STATUS")
    print("="*60)
    print(json.dumps(orchestrator.status(), indent=2))