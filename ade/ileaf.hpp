///
/// ileaf.hpp
/// ade
///
/// Purpose:
/// Define leafs for tensor equation graph
///

#include "ade/itensor.hpp"
#include "ade/idata.hpp"

#ifndef ADE_ILEAF_HPP
#define ADE_ILEAF_HPP

namespace ade
{

/// Leaf of the graph commonly representing the variable in an equation
struct iLeaf : public iTensor, public iData
{
	virtual ~iLeaf (void) = default;

	/// Implementation of iTensor
	void accept (iTraveler& visiter) override
	{
		visiter.visit(this);
	}
};

/// Leaf smart pointer
using LeafptrT = std::shared_ptr<iLeaf>;

}

#endif // ADE_ILEAF_HPP
