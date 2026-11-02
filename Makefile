.PHONY: setup train clean

setup:
	pip install --upgrade pip
	pip install -r requirements.txt

train:
	python train.py --model all

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
