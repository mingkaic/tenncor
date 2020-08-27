
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "dbg/compare/equal.hpp"

#include "tenncor/tenncor.hpp"

namespace py = pybind11;

PYBIND11_MODULE(compare, m)
{
	m.doc() = "compare teq graphs";

	m
		// ==== to stdout functions ====
		.def("is_equal",
		[](const eteq::ETensor& l, const eteq::ETensor& r)
		{
			return is_equal(l, r);
		},
		"Return true if roots are structurally equal")
		.def("percent_dataeq",
		[](const eteq::ETensor& l, const eteq::ETensor& r)
		{
			return percent_dataeq(l, r);
		});
}
