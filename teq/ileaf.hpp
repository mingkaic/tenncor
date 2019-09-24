///
/// ileaf.hpp
/// teq
///
/// Purpose:
/// Define leafs for tensor equation graph
///

#include "teq/itensor.hpp"
#include "teq/idata.hpp"

#ifndef TEQ_ILEAF_HPP
#define TEQ_ILEAF_HPP

namespace teq
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

	/// Return true if leaf is immutable, otherwise false
	virtual bool is_const (void) const = 0;
};

/// Leaf smart pointer
using LeafptrT = std::shared_ptr<iLeaf>;

}

#endif // TEQ_ILEAF_HPP
