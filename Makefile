PYTHON ?= python3

.PHONY: all lint test simulate paper
all: lint test simulate paper

lint:
	$(PYTHON) -m black --check sim tests
	$(PYTHON) -m isort --check-only sim tests
	$(PYTHON) -m flake8 sim tests

test:
	$(PYTHON) -m pytest -q

simulate:
	$(PYTHON) sim/coverage_simulation.py

paper:
	cd note && latexmk -pdf -interaction=nonstopmode -halt-on-error inferred_compliance_note.tex
