from setuptools import setup, find_packages, Extension

def readme():
    with open("README.md") as f:
        return f.read()

setup(
    name = "tenncor",
    version = "0.0.3",
    description = "Tensor algebra module.",
    long_description = readme(),
    long_description_content_type ="text/markdown",
    classifiers = [
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
    keywords = "",
    url = "https://github.com/mingkaic/tenncor",
    author = "Ming Kai Chen",
    author_email = "mingkaichen2009@gmail.com",
    license = "MIT",
    packages = find_packages(),
    ext_modules=[Extension('', [], libraries=['tenncor/tenncor.so'])],
    install_requires = [],
    test_suite = "",
    tests_require = [],
    zip_safe = False,
    python_requires = '>=3.6',
)
