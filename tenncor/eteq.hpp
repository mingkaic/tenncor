///
/// derive.hpp
/// tenncor
///
/// Purpose:
/// Implement eteq gradient definition for supported operations
///

#ifndef TENNCOR_ETEQ_HPP
#define TENNCOR_ETEQ_HPP

#include "tenncor/eteq/eteq.hpp"

#include "tenncor/distrib.hpp"

namespace tcr
{

eteq::ETensorsT derive_with_manager (
	distr::iDistrManager& mgr,
	eteq::ETensor root, const eteq::ETensorsT& targets);

/// Derive root with respect to target and optimized
eteq::ETensorsT derive (eteq::ETensor root, const eteq::ETensorsT& targets);

}

#endif // TENNCOR_ETEQ_HPP
