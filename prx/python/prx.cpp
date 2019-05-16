
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"

#include "ead/generated/pyapi.hpp"

#include "prx/api.hpp"

namespace py = pybind11;

PYBIND11_MODULE(prx, m)
{
	m.doc() = "prx variables";

	// ==== node ====
	py::object node = (py::object)
		py::module::import("ead.age").attr("NodeptrT<PybindT>");

	m.def("fully_connected", &prx::fully_connect<PybindT>,
		"ead::NodeptrT<T> fully_connected (ead::NodesT<T> inputs, "
		"ead::NodesT<T> weights, ead::NodeptrT<T> bias)",
		py::arg("inputs"), py::arg("weights"), py::arg("bias") = nullptr);

		m.def("conv2d", &prx::conv2d<PybindT>,
		"ead::NodeptrT<T> conv2d (ead::NodeptrT<T> image, "
		"ead::NodeptrT<T> kernel, ead::NodeptrT<T> bias)",
		py::arg("image"), py::arg("kernel"), py::arg("bias") = nullptr);
}
