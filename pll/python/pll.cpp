#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"

#include "ead/generated/pyapi.hpp"
#include "ead/parse.hpp"

#include "pll/session.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pll, m)
{
	m.doc() = "pll session";

	// ==== session ====
	auto isess = (py::class_<ead::iSession>)
		py::module::import("ead.ead").attr("iSession");
	py::class_<pll::Session> session(m, "Session", isess);

	py::implicitly_convertible<ead::iSession,pll::Session>();
	session
		.def(py::init<int,pll::OpWeightT>(),
			py::arg("nthread") = 2,
			py::arg("weights") = pll::OpWeightT())
		.def("optimize",
			[](py::object self, std::string filename)
			{
				auto sess = self.cast<pll::Session*>();
				opt::OptCtx rules = ead::parse_file<PybindT>(filename);
				sess->optimize(rules);
			},
			py::arg("filename") = "cfg/optimizations.rules",
			"Optimize using rules for specified filename");
}
