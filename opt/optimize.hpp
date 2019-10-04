///
/// optimize.hpp
/// opt
///
/// Purpose:
/// Implement algorithm that applies conversion rules to graph roots
///

#include "opt/matcher.hpp"
#include "opt/iconverter.hpp"

#ifndef OPT_OPTIMIZE_HPP
#define OPT_OPTIMIZE_HPP

namespace opt
{

/// Function that takes raw tensor pointer and cast to constant tensor
/// If input tensor is not constant return null
using CstConvertF = std::function<teq::TensptrT(teq::iTensor*)>;

/// Encapsulation of all conversion rules
struct OptCtx
{
	/// Voters for identifying subgraphs associated with conversion target ids
	VoterPool voters_;

	/// Function for identifying constant tensors
	CstConvertF const_conv_;

	/// Map of conversion target ids to conversion target builders
	std::unordered_map<std::string,ConvptrT> converts_;
};

/// Return optimized roots where optimization rules are applied to subgraphs
/// Optimized graph roots are moved back to their corresponding root tensors
/// Additionally two or more tensors sharing symbolically identical
/// representations are "joined" (with the exception of tensors in roots set)
teq::TensptrsT optimize (teq::TensptrsT roots, const OptCtx& opts);

}

#endif // OPT_OPTIMIZE_HPP
