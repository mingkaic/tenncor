from conans import ConanFile, CMake
import os

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
        "cppkg/0.1@mingkaic-co/stable",
        "Ppconsul/0.2.1@mingkaic-co/stable",
        "g3log/1.3.3",
        "eigen/3.3.8",
        "pybind11/2.6.1",
    )
    generators = "cmake", "cmake_find_package_multi"

    def source(self):
        self.run("git clone {}.git .".format(self.url))

    def configure(self):
        g3log_options = self.options["g3log"]
        g3log_options.use_dynamic_logging_levels = True
        g3log_options.change_debug_to_dbug = True
        g3log_options.shared = False

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        self.copy("*.hpp", dst=os.path.join("include", "tenncor"), src="tenncor")
        self.copy("*.lib", dst="lib", keep_path=False)
        self.copy("*.dll", dst="bin", keep_path=False)
        self.copy("*.so", dst="lib", keep_path=False)
        self.copy("*.dylib", dst="lib", keep_path=False)
        self.copy("*.a", dst="lib", keep_path=False)

    def package_info(self):
        self.cpp_info.libs = ["{}_{}".format(self.name, module) for module in _modules]
