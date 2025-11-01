.PHONY: setup

setup:
	pip install -e .
	python -m tessrax.selftest
