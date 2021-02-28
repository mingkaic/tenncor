#include <fstream>
#include <sstream>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "dbg/profile/profile.hpp"

#include "tenncor/tenncor.hpp"

namespace py = pybind11;

PYBIND11_MODULE(tenncor_profile, m)
{
	m.doc() = "profile teq graphs";

	m
		// ==== to stdout functions ====
		.def("profile",
		[](const std::string& filename, eteq::ETensorsT roots)
		{
            teq::TensT rs;
            rs.reserve(roots.size());
            std::transform(roots.begin(), roots.end(), std::back_inserter(rs),
            [](eteq::ETensor etens) { return etens.get(); });
            dbg::profile::gexf_write(filename, rs);
		},
		"Profile graph of tensors",
		py::arg("filename"),
		py::arg("roots"));
}
