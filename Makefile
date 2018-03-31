TEST := bazel test --test_output=all --cache_test_results=no

all: unittest accepttest

unittest: tensortest graphtest operatetest

tensortest:
	$(TEST) //tests/unit:test_tensor

graphtest:
	$(TEST) //tests/unit:test_graph

operatetest:
	$(TEST) //tests/unit:test_operate

accepttest:
	$(TEST) //tests/regress:test

acceptdata: cleandata
	python tests/regress/tf_generate/tf_generate.py

fmt:
	astyle --project --recursive --suffix=none *.hpp,*.ipp,*.cpp

cleandata:
	rm -f tests/regress/samples/*
