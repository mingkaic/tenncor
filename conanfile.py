import os
import sys

from conans import ConanFile, CMake, tools

class TenncorConan(ConanFile):
    name = "tenncor"
    version = "0.1"
    license = "MIT"
    author = "Ming Kai Chen <mingkaichen2009@gmail.com>"
    url = "https://github.com/mingkaic/tenncor"
    description = "Tensor differentiation package"
    topics = ["conan", "machine learning"]
    settings = "os", "compiler", "build_type", "arch"
    requires = (
        "cppkg/0.1.1@mingkaic-co/stable",
        "Ppconsul/0.2.1@mingkaic-co/stable",
        "g3log/1.3.3",
        "eigen/3.3.7",
        "pybind11/2.6.0",
    )
    generators = "cmake", "cmake_find_package_multi"

    def _configure_cmake(self):
        cmake = CMake(self)
        cmake.definitions["PYTHON_EXECUTABLE"] = sys.executable
        cmake.definitions["CMAKE_BUILD_TYPE"] = self.settings.build_type
        cmake.configure()
        return cmake

    def configure(self):
        if self.settings.os == "Windows" and self.settings.compiler == "Visual Studio" and tools.Version(self.settings.compiler.version) < 14:
            raise ConanInvalidConfiguration("gRPC can only be built with Visual Studio 2015 or higher.")

        g3log_options = self.options["g3log"]
        g3log_options.use_dynamic_logging_levels = True
        g3log_options.change_debug_to_dbug = True
        g3log_options.shared = False

    def source(self):
        self.run("git clone {}.git .".format(self.url))

        os.remove("dbg/peval/emit/gemitter.pb.h")
        os.remove("dbg/peval/emit/gemitter.pb.cc")
        os.remove("internal/onnx/onnx.pb.h")
        os.remove("internal/onnx/onnx.pb.cc")
        os.remove("tenncor/serial/oxsvc/distr.ox.pb.h")
        os.remove("tenncor/serial/oxsvc/distr.ox.pb.cc")

    def build(self):
        cmake = self._configure_cmake()
        cmake.build()

    def package(self):
        self.copy(pattern="LICENSE.*", dst="licenses", keep_path=False)
        cmake = self._configure_cmake()
        cmake.install()

    def package_info(self):
        self.cpp_info.names["cmake_find_package"] = self.name
        self.cpp_info.names["cmake_find_package_multi"] = self.name
        self.cpp_info.libs = ["ctenncor"]
