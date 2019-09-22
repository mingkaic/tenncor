#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"

#include "eteq/generated/pyapi.hpp"
#include "eteq/parse.hpp"

#include "pll/session.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pll, m)
{
	m.doc() = "pll session";

	// ==== session ====
	auto isess = (py::class_<eteq::iSession>)
		py::module::import("eteq.eteq").attr("iSession");
	py::class_<pll::Session> session(m, "Session", isess);

	py::implicitly_convertible<eteq::iSession,pll::Session>();
	session
		.def(py::init<int,pll::OpWeightT>(),
			py::arg("nthread") = 2,
			py::arg("weights") = pll::OpWeightT())
		.def("optimize",
			[](py::object self, std::string filename)
			{
				auto sess = self.cast<pll::Session*>();
				opt::OptCtx rules = eteq::parse_file<PybindT>(filename);
				sess->optimize(rules);
			},
			py::arg("filename") = "cfg/optimizations.rules",
			"Optimize using rules for specified filename");
}
