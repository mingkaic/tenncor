#include "python/eteq_ext.hpp"
#include "python/layr_ext.hpp"
#include "python/query_ext.hpp"
#include "python/distrib_ext.hpp"

#define CUSTOM_PYBIND_EXT(MODULE)\
eteq_ext(MODULE);\
layr_ext(MODULE);\
query_ext(MODULE);\
distrib_ext(MODULE);
