
#ifndef PYTHON_LAYR_EXT_HPP
#define PYTHON_LAYR_EXT_HPP

#include <sstream>

#include "pybind11/functional.h"

#include "tenncor/generated/pyapi.hpp"

#include "tenncor/trainer/trainer.hpp"

#include "tenncor/python/tenncor_utils.hpp"

void layr_ext(py::module& m);

#endif // PYTHON_LAYR_EXT_HPP
