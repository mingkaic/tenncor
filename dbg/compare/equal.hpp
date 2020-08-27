
#ifndef DBG_GRAPHEQ_HPP
#define DBG_GRAPHEQ_HPP

#include "tenncor/eteq/eteq.hpp"

#include "tenncor/opt/opt.hpp"

/// Return true if lroot and rroot graphs are structurally equal
bool is_equal (const eteq::ETensor& lroot, const eteq::ETensor& rroot);

/// Return true if lroot and rroot graphs have the same data
bool is_dataeq (const eteq::ETensor& lroot, const eteq::ETensor& rroot);

/// Return percent of nodes that are data equivalent
double percent_dataeq (const eteq::ETensor& lroot, const eteq::ETensor& rroot);

#endif // DBG_GRAPHEQ_HPP
