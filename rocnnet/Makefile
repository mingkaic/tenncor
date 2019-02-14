
py_build:
	bazel build //:rocnnet_py

py_export: py_build
	cp -f bazel-bin/rocnnet.so notebooks/
	cp -f bazel-bin/external/com_github_mingkaic_tenncor/llo/*.so notebooks/llo
