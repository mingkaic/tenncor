#include <fstream>
#include <sstream>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "dbg/print/print.hpp"

#include "tenncor/tenncor.hpp"

namespace py = pybind11;

PYBIND11_MODULE(print, m)
{
	m.doc() = "print teq graphs to stream";

	m
		// ==== to stdout functions ====
		.def("print_graph",
		[](eteq::ETensor root, bool showshape)
		{
			PrettyEquation peq;
			peq.cfg_.showshape_ = showshape;
			peq.print(std::cout, root);
		},
		"Print graph of root tensor to stdout",
		py::arg("root"),
		py::arg("showshape") = false)
		.def("print_graphcsv",
		[](eteq::ETensor root, bool showshape)
		{
			CSVEquation ceq;
			ceq.showshape_ = showshape;
			root->accept(ceq);
			ceq.to_stream(std::cout);
		},
		"Print csv of graph edges to stdout",
		py::arg("root"),
		py::arg("showshape") = false)

		// ==== to string functions ====
		.def("graph_to_str",
		[](eteq::ETensor root, bool showshape)
		{
			std::stringstream ss;
			PrettyEquation peq;
			peq.cfg_.showshape_ = showshape;
			peq.print(ss, root);
			return ss.str();
		},
		"Return graph of root tensor as string",
		py::arg("root"),
		py::arg("showshape") = false)
		.def("graph_to_csvstr",
		[](eteq::ETensor root, bool showshape)
		{
			std::stringstream ss;
			CSVEquation ceq;
			ceq.showshape_ = showshape;
			root->accept(ceq);
			ceq.to_stream(ss);
			return ss.str();
		},
		"Return csv of graph edges as string",
		py::arg("root"),
		py::arg("showshape") = false)
		.def("multigraph_to_csvstr",
		[](eteq::ETensorsT roots, bool showshape)
		{
			std::stringstream ss;
			CSVEquation ceq;
			ceq.showshape_ = showshape;
			for (auto& root : roots)
			{
				root->accept(ceq);
			}
			ceq.to_stream(ss);
			return ss.str();
		},
		"Return csv of graph edges of multiple roots as string",
		py::arg("roots"),
		py::arg("showshape") = false)

		// ==== to file functions ====
		.def("graph_to_file",
		[](eteq::ETensor root, std::string filename, bool showshape)
		{
			std::ofstream outstr(filename);
			if (outstr.is_open())
			{
				PrettyEquation peq;
				peq.cfg_.showshape_ = showshape;
				peq.print(outstr, root);
			}
			else
			{
				global::warnf("failed to print graph to file `%s`",
					filename.c_str());
			}
		},
		"Stream graph of root tensor to file",
		py::arg("root"),
		py::arg("filename"),
		py::arg("showshape") = false)
		.def("graph_to_csvfile",
		[](eteq::ETensor root, std::string filename, bool showshape)
		{
			std::ofstream outstr(filename);
			if (outstr.is_open())
			{
				CSVEquation ceq;
				ceq.showshape_ = showshape;
				root->accept(ceq);
				ceq.to_stream(outstr);
			}
			else
			{
				global::warnf("failed to write csv to file `%s`", filename.c_str());
			}
		},
		"Stream csv of graph edges to file",
		py::arg("root"),
		py::arg("filename"),
		py::arg("showshape") = false)
		.def("multigraph_to_csvfile",
		[](eteq::ETensorsT roots, std::string filename, bool showshape)
		{
			std::ofstream outstr(filename);
			if (outstr.is_open())
			{
				CSVEquation ceq;
				ceq.showshape_ = showshape;
				for (auto& root : roots)
				{
					root->accept(ceq);
				}
				ceq.to_stream(outstr);
			}
			else
			{
				global::warnf("failed to write csv to file `%s`", filename.c_str());
			}
		},
		"Return csv of graph edges of multiple roots to file",
		py::arg("roots"),
		py::arg("filename"),
		py::arg("showshape") = false);
}
