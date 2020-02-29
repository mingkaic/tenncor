
#ifndef PYTHON_ETEQ_EXT_HPP
#define PYTHON_ETEQ_EXT_HPP

#include "pybind11/stl.h"

#include "teq/logs.hpp"

#include "pyutils/convert.hpp"

#include "eigen/device.hpp"
#include "eigen/random.hpp"

#include "eteq/derive.hpp"
#include "eteq/layer.hpp"
#include "eteq/optimize.hpp"

#include "generated/pyapi.hpp"

namespace py = pybind11;

namespace pyead
{

using ETensT = eteq::ETensor<PybindT>;

using ETensPairT = std::pair<ETensT,ETensT>;

template <typename T>
py::array typedata_to_array (teq::iTensor& tens, py::dtype dtype)
{
	assert(egen::get_type<PybindT>() == tens.type_code());
	auto pshape = pyutils::c2pshape(tens.shape());
	return py::array(dtype,
		py::array::ShapeContainer(pshape.begin(), pshape.end()),
		tens.device().data());
}

}

void eteq_ext (py::module& m);

#endif // PYTHON_ETEQ_EXT_HPP
