#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"

#include "internal/global/global.hpp"

#include "ccur/session.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ccur, m)
{
	m.doc() = "ccur session";

	// ==== session ====
	auto isess = (py::class_<teq::iSession>)
		py::module::import("eteq.eteq").attr("iSession");
	py::class_<ccur::Session> session(m, "Session", isess);

	py::implicitly_convertible<teq::iSession,ccur::Session>();
	session
		.def(py::init<int,ccur::OpWeightT>(),
			py::arg("nthread") = 2,
			py::arg("weights") = ccur::OpWeightT());
}
