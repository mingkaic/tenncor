///
/// approx.hpp
/// layr
///
/// Purpose:
/// Define error approximation algorithms and variable assignment utilities
///

#include "eteq/make.hpp"

#ifndef LAYR_APPROX_HPP
#define LAYR_APPROX_HPP

namespace layr
{

template <typename T>
struct EVarHasher
{
	size_t operator ()(const eteq::EVariable<T>& evar) const
	{
		return std::hash<void*>()(evar.get());
	}
};

/// Ordered association between variable and error
template <typename T>
using VarMapT = std::unordered_map<eteq::EVariable<T>,eteq::ETensor<T>,EVarHasher<T>>;

/// Function that returns the error between two nodes,
/// left node contains expected values, right contains resulting values
template <typename T>
using ErrorF = std::function<eteq::ETensor<T>(const eteq::ETensor<T>&,const eteq::ETensor<T>&)>;

/// Function that approximate error of sources
/// given a vector of variables and its corresponding errors
template <typename T>
using ApproxF = std::function<VarMapT<T>(const eteq::ETensor<T>&,const eteq::EVariablesT<T>&)>;

}

#endif // LAYR_APPROX_HPP
