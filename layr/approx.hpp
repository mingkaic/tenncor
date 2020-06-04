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

/// Association between variable and error root
template <typename T>
using VarErrT = std::pair<eteq::EVariable<T>,eteq::ETensor<T>>;

/// Ordered association between variable and error
template <typename T>
using VarErrsT = std::vector<VarErrT<T>>;

/// Function that returns error given two tensor inputs
template <typename T>
using BErrorF = std::function<eteq::ETensor<T>(const eteq::ETensor<T>&,const eteq::ETensor<T>&)>;

/// Function that returns the error given list of tensor inputs
template <typename T>
using ErrorF = std::function<eteq::ETensor<T>(const eteq::ETensorsT<T>&)>;

/// Function that approximate error of sources
/// given a vector of variables and its corresponding errors
template <typename T>
using ApproxF = std::function<VarErrsT<T>(const eteq::ETensor<T>&,const eteq::EVariablesT<T>&)>;

}

#endif // LAYR_APPROX_HPP
