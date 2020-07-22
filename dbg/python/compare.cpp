
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "teq/logs.hpp"

#include "dbg/compare/equal.hpp"

#include "tenncor/tenncor.hpp"

namespace py = pybind11;

PYBIND11_MODULE(compare, m)
{
	m.doc() = "compare teq graphs";

	m
		// ==== to stdout functions ====
		.def("is_equal",
		[](const eteq::ETensor<PybindT>& l, const eteq::ETensor<PybindT>& r)
		{
			return is_equal<PybindT>(l, r);
		},
		"Return true if roots are structurally equal")
		.def("percent_dataeq",
		[](const eteq::ETensor<PybindT>& l, const eteq::ETensor<PybindT>& r)
		{
			return percent_dataeq<PybindT>(l, r);
		});
}
