
#ifndef PYTHON_ETEQ_EXT_HPP
#define PYTHON_ETEQ_EXT_HPP

#include "pybind11/stl.h"

#include "pyutils/convert.hpp"

#include "eigen/eigen.hpp"

#include "eteq/eteq.hpp"

#include "generated/pyapi.hpp"

namespace py = pybind11;

namespace pyeteq
{

using ETensT = eteq::ETensor<PybindT>;

using ETensorsT = eteq::ETensorsT<PybindT>;

using ETensPairT = std::pair<ETensT,ETensT>;

template <typename T>
py::array typedata_to_array (T* data, teq::Shape shape,
	size_t typecode, py::dtype dtype)
{
	assert(egen::get_type<PybindT>() == typecode);
	auto pshape = pyutils::c2pshape(shape);
	return py::array(dtype, py::array::ShapeContainer(
		pshape.begin(), pshape.end()), data);
}

}

void eteq_ext (py::module& m);

#endif // PYTHON_ETEQ_EXT_HPP
