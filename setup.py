import os
import subprocess

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, source_dir=""):
        Extension.__init__(self, name, sources=[])
        self.source_dir = os.path.abspath(source_dir)

class ConanCMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake missing - probably upgrade a newer version of pip")
        try:
            subprocess.check_output(["conan", "--version"])
        except OSError:
            raise RuntimeError("Conan missing")

        super().run()

    def build_extension(self, ext):
        ext_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        if not ext_dir.endswith(os.path.sep):
            ext_dir += os.path.sep

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(["conan", "install", "-if", ".", ext.source_dir], cwd=self.build_temp)
        subprocess.check_call(["conan", "build", "-bf", ".", ext.source_dir], cwd=self.build_temp)

__version__ = "0.0.3"

def readme():
    with open("README.md") as f:
        return f.read()

setup(
    name = "tenncor",
    version = __version__,
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
    install_requires = [],
    test_suite = "",
    tests_require = [],
    zip_safe = False,
    python_requires = '>=3.6',
    ext_modules=[CMakeExtension("tenncor")],
    cmdclass={"build_ext": ConanCMakeBuild},
)
