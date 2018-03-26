all: clean
	python tests/accept/tf_generate/tf_generate.py

clean:
	rm -f tests/accept/samples/*
