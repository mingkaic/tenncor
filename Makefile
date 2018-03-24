all: clean
	python tests/regression/tf_generate/tf_generate.py

clean:
	rm -f tests/regression/samples/*
