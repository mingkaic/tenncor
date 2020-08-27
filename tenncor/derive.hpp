///
/// derive.hpp
/// tenncor
///
/// Purpose:
/// Implement eteq gradient definition for supported operations
///

#ifndef TENNCOR_DERIVE_HPP
#define TENNCOR_DERIVE_HPP

#include "tenncor/distr.hpp"

namespace tcr
{

eteq::ETensorsT derive_with_manager (
	distr::iDistrManager& mgr,
	eteq::ETensor root, const eteq::ETensorsT& targets);

/// Derive root with respect to target and optimized
eteq::ETensorsT derive (eteq::ETensor root, const eteq::ETensorsT& targets);

}

#endif // TENNCOR_DERIVE_HPP
