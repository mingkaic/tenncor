TEST := bazel test --test_output=all

all: unittest accepttest

unittest: tensortest graphtest operatetest

tensortest:
	$(TEST) //tests/unit:test_tensor

graphtest:
	$(TEST) //tests/unit:test_graph

operatetest:
	$(TEST) //tests/unit:test_operate

accepttest:
	$(TEST) //tests/accept:test

acceptdata: cleandata
	python tests/accept/tf_generate/tf_generate.py

fmt:
	astyle --project --recursive --suffix=none *.hpp,*.ipp,*.cpp

cleandata:
	rm -f tests/accept/samples/*
