#include "python/eteq_ext.hpp"
#include "python/layr_ext.hpp"

#define CUSTOM_PYBIND_EXT(MODULE)\
eteq_ext(MODULE);\
layr_ext(MODULE);