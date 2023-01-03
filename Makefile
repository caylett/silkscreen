hello:
	echo "Hello, World!"

clean: 
	rm -rf .virtualenv requirements.txt

setup:
	python3 -m venv --prompt dev .virtualenv && source .virtualenv/bin/activate && pip3 install -U pip pip-tools && pip-compile requirements.in && pip-sync requirements.txt