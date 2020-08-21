
#include <sstream>

#include "pybind11/functional.h"

#include "tenncor/trainer/apply_update.hpp"
#include "tenncor/trainer/dbn.hpp"

#include "python/tenncor_utils.hpp"

#ifndef PYTHON_LAYR_EXT_HPP
#define PYTHON_LAYR_EXT_HPP

void layr_ext(py::module& m);

#endif // PYTHON_LAYR_EXT_HPP
