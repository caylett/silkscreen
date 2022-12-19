hello:
	echo "Hello, World!"

clean: 
	rm -rf .venv requirements.txt

setup:
	python3 -m venv --prompt dev .venv
	pip3 install -U pip pip-tools
	pip-compile requirements.in
	pip-sync