///
/// ileaf.hpp
/// ade
///
/// Purpose:
/// Define interfaces and building blocks for an equation graph
///

#include "ade/itensor.hpp"

#ifndef ADE_TENSOR_HPP
#define ADE_TENSOR_HPP

namespace ade
{

/// Leaf of the graph commonly representing the variable in an equation
struct iLeaf : public iTensor
{
	virtual ~iLeaf (void) = default;

	/// Implementation of iTensor
	void accept (iTraveler& visiter) override
	{
		visiter.visit(this);
	}

	/// Return pointer to internal data
	virtual void* data (void) = 0;

	/// Return const pointer to internal data
	virtual const void* data (void) const = 0;

	/// Return data type encoding
	virtual size_t type_code (void) const = 0;

	/// Return data type label (for better readability)
	virtual std::string type_label (void) const = 0;
};

/// Leaf smart pointer
using LeafptrT = std::shared_ptr<iLeaf>;

}

#endif // ADE_TENSOR_HPP
