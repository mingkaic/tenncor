///
/// itensor.hpp
/// teq
///
/// Purpose:
/// Define interfaces and building blocks for an equation graph
///

#include <unordered_set>

#include "teq/shape.hpp"

#ifndef TEQ_INTERFACE_HPP
#define TEQ_INTERFACE_HPP

namespace teq
{

struct iLeaf;

struct iFunctor;

struct Placeholder;

/// Interface to travel through graph, treating iLeaf and iFunctor differently
struct iTraveler
{
	virtual ~iTraveler (void) = default;

	/// Visit leaf node
	virtual void visit (iLeaf& leaf) = 0;

	/// Visit functor node
	virtual void visit (iFunctor& func) = 0;

	/// Visit placeholder node
	virtual void visit (Placeholder& placeholder) = 0;
};

/// Interface of traversible and differentiable nodes with shape information
struct iTensor
{
	virtual ~iTensor (void) = default;

	iTensor* clone (void) const
	{
		return this->clone_impl();
	}

	/// Obtain concrete information on either leaf or functor implementations
	virtual void accept (iTraveler& visiter) = 0;

	/// Return the shape held by this tensor
	virtual Shape shape (void) const = 0;

	/// Return the string representation of the tensor
	virtual std::string to_string (void) const = 0;

protected:
	virtual iTensor* clone_impl (void) const = 0;
};

/// Tensor smart pointer
using TensptrT = std::shared_ptr<iTensor>;

/// Tensor weak pointers
using TensrefT = std::weak_ptr<iTensor>;

/// Vector of raw tensor pointers
using TensT = std::vector<iTensor*>;

/// Vector of tensor smart pointers
using TensptrsT = std::vector<TensptrT>;

/// Hash set of raw tensor pointers
using TensSetT = std::unordered_set<iTensor*>;

/// Hash set of tensor smart pointers
using TensptrSetT = std::unordered_set<TensptrT>;

}

#endif // TEQ_INTERFACE_HPP
