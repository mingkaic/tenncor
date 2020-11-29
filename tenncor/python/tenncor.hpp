#include "tenncor/python/eteq_ext.hpp"
#include "tenncor/python/layr_ext.hpp"
#include "tenncor/python/query_ext.hpp"
#include "tenncor/python/distr_ext.hpp"

#define CUSTOM_PYBIND_EXT(MODULE)\
eteq_ext(MODULE);\
layr_ext(MODULE);\
query_ext(MODULE);\
distr_ext(MODULE);
