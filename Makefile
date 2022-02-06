PYTHON=3.8
BASENAME=$(shell basename $(CURDIR))

env:
	conda create -n $(BASENAME)  python=$(PYTHON)

setup:
	conda install --file requirements.txt -c conda-forge -c pytorch
	pip install -r requirements-pip.txt
	pre-commit install
	mypy --install-types

format:
	black .
	isort .

lint:
	pytest src --flake8 --pylint --mypy

utest:
	PYTHONPATH=src pytest test/utest --cov=src --cov-report=html --cov-report=term --cov-config=setup.cfg

cov:
	open htmlcov/index.html

serving:
	PYTHONPATH=src uvicorn src.backend:app --reload

frontend:
	PYTHONPATH=src streamlit run src/frontend.py
