
#ifndef PYTHON_QUERY_EXT_HPP
#define PYTHON_QUERY_EXT_HPP

#include <sstream>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "teq/teq.hpp"

#include "eteq/eteq.hpp"

#include "query/query.hpp"

#include "generated/pyapi.hpp"

namespace py = pybind11;

namespace pyquery
{

struct Statement final
{
	Statement (teq::TensptrsT tens) : tracked_(tens)
	{
		for (auto ten : tens)
		{
			ten->accept(sindex_);
		}
	}

	query::Query sindex_;

	teq::TensptrsT tracked_;
};

}

void query_ext(py::module& m);

#endif // PYTHON_QUERY_EXT_HPP
