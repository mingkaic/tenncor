
#ifndef PYTHON_TENNCOR_UTIL_HPP
#define PYTHON_TENNCOR_UTIL_HPP

#include "pybind11/stl.h"

#include "tenncor/pyutils/convert.hpp"

#include "tenncor/tenncor.hpp"

namespace py = pybind11;

namespace pytenncor
{

using ETensPairT = std::pair<eteq::ETensor,eteq::ETensor>;

template <typename T>
py::array typedata_to_array (T* data, teq::Shape shape,
	size_t typecode, py::dtype dtype)
{
	assert(egen::get_type<PybindT>() == typecode);
	auto pshape = pyutils::c2pshape(shape);
	return py::array(dtype, py::array::ShapeContainer(
		pshape.begin(), pshape.end()), data);
}

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

#endif // PYTHON_TENNCOR_UTIL_HPP
