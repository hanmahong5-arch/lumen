.PHONY: test lint build publish clean install

test:
	python -m pytest tests/ -v

lint:
	python -m py_compile lumen/__init__.py
	python -m py_compile lumen/pricing.py
	python -m py_compile lumen/cost.py
	python -m py_compile lumen/replay.py
	python -m py_compile lumen/agent.py
	python -m py_compile lumen/trace_reader.py
	python -m py_compile lumen/integrations/langgraph.py
	python -m py_compile lumen/integrations/langgraph_tracer.py

install:
	pip install -e .

install-dev:
	pip install -e ".[langgraph]"
	pip install pytest build twine

build: clean
	python -m build

publish: build
	python -m twine upload dist/*

publish-test: build
	python -m twine upload --repository testpypi dist/*

clean:
	rm -rf dist/ build/ *.egg-info lumen/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
