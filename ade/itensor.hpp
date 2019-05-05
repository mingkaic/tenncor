///
/// itensor.hpp
/// ade
///
/// Purpose:
/// Define interfaces and building blocks for an equation graph
///

#include "ade/shape.hpp"

#ifndef ADE_INTERFACE_HPP
#define ADE_INTERFACE_HPP

namespace ade
{

struct iLeaf;

struct iFunctor;

/// Interface to travel through graph, treating iLeaf and iFunctor differently
struct iTraveler
{
	virtual ~iTraveler (void) = default;

	/// Visit leaf node
	virtual void visit (iLeaf* leaf) = 0;

	/// Visit functor node
	virtual void visit (iFunctor* func) = 0;
};

/// Interface of traversible and differentiable nodes with shape information
struct iTensor
{
	virtual ~iTensor (void) = default;

	/// Obtain concrete information on either leaf or functor implementations
	virtual void accept (iTraveler& visiter) = 0;

	/// Return the shape held by this tensor
	virtual const Shape& shape (void) const = 0;

	/// Return the string representation of the tensor
	virtual std::string to_string (void) const = 0;
};

/// Tensor smart pointer
using TensptrT = std::shared_ptr<iTensor>;

/// Tensor weak pointers
using TensrefT = std::weak_ptr<iTensor>;

}

#endif // ADE_INTERFACE_HPP
