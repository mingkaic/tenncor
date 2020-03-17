
#ifndef PYTHON_LAYR_EXT_HPP
#define PYTHON_LAYR_EXT_HPP

#include <fstream>
#include <sstream>

#include "pybind11/stl.h"
#include "pybind11/functional.h"

#include "pyutils/convert.hpp"

#include "teq/logs.hpp"

#include "eigen/device.hpp"

#include "eteq/eteq.hpp"

#include "trainer/sgd.hpp"
#include "trainer/dqn.hpp"
#include "trainer/dbn.hpp"

#include "generated/pyapi.hpp"

namespace py = pybind11;

namespace pylayr
{

using VPairT = std::pair<eteq::EVariable<PybindT>,eteq::ETensor<PybindT>>;

using VPairsT = std::vector<VPairT>;

using ApproxF = std::function<VPairsT(const VPairsT&)>;

layr::VarMapT<PybindT> convert (const VPairsT& op);

VPairsT convert (const layr::VarMapT<PybindT>& op);

layr::ApproxF<PybindT> convert (ApproxF f);

}

void layr_ext(py::module& m);

#endif // PYTHON_LAYR_EXT_HPP