all: clean
	python tests/functional/tf_generate/tf_generate.py

clean:
	rm -f tests/functional/samples/*
