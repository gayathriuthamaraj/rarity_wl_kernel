.PHONY: install verify inspect
install:
	python -m pip install -e ".[dev,graphs]"

verify:
	python scripts/verify_dataset.py

inspect:
	python scripts/inspect_dataset.py
