# Tessrax Quick-Start Guide

Run a complete contradiction metabolism cycle in seconds.

```bash
git clone https://github.com/joshuavetos/Tessrax
cd Tessrax
pip install -r requirements.txt
python -m core.engine

What happens
	1.	Generates sample agent claims.
	2.	Runs contradiction_engine.run_contradiction_cycle()
	3.	Routes result through governance_kernel.route()
	4.	Appends an auditable record to data/governance_ledger.jsonl.

To disable the auto-demo during CI:
export QUICKSTART=0