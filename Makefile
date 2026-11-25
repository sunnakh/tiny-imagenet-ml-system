PYTHON := .venv/bin/python
PROFILE ?= fixture
STATE ?= .state/$(PROFILE)
ARGS ?=

.PHONY: setup demo check
setup:
	test -x $(PYTHON) || uv venv --python 3.12
	uv pip install --python $(PYTHON) -e '.[dev]'

demo: setup
	$(PYTHON) -m tiny_imagenet --state "$(STATE)" demo --profile "$(PROFILE)" $(ARGS)

check:
	$(PYTHON) -m ruff check tiny_imagenet tests
	$(PYTHON) -m ruff format --check tiny_imagenet tests
	$(PYTHON) -m mypy
	$(PYTHON) -m unittest discover -s tests -v
