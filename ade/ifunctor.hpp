///
///	functor.hpp
///	ade
///
///	Purpose:
///	Define functor nodes of an equation graph
///

#include <algorithm>
#include <cassert>
#include <list>
#include <unordered_map>

#include "ade/tensor.hpp"

#ifndef ADE_IFUNCTOR_HPP
#define ADE_IFUNCTOR_HPP

namespace ade
{

/// Type of functor arguments
using ArgsT = std::vector<MappedTensor>;

struct iOperation
{
	virtual ~iOperation (void) = default;

	virtual std::string to_string (void) const = 0;

	virtual size_t opnum (void) const = 0;

	/// Let A = this operation, X = child operation
	/// Return derivative of this operation on args children with respect to
	/// the child at index gradidx (return dA/dX)
	virtual Tensorptr gradient (ArgsT args, size_t gradidx) const = 0;

	/// Let F = parent operation, A = this operation, X = child operation
	/// Return dF/dX, given wrt_me = dF/dA, wrt_child = dA/dX, the wrt_me
	/// carries a jacobian which mapps A-coordinate to X-coordinate system
	virtual Tensorptr chain_grad (Tensorptr& wrt_child,
		MappedTensor wrt_me) const = 0;

	/// Let F = parent operation, X = child operation
	/// Return sum of dF/dX
	virtual Tensorptr add_grads (ArgsT& grads) const = 0;
};

using OpPtrT = std::shared_ptr<iOperation>;

/// Interface of iOperation-defined operation node
struct iFunctor : public iTensor
{
	virtual ~iFunctor (void) = default;

	/// Implementation of iTensor
	void accept (iTraveler& visiter) override
	{
		visiter.visit(this);
	}

	/// Return OPCODE mapping to forward and gradient operators
	virtual const iOperation& get_code (void) const = 0;

	/// Return children nodes as a vector of raw pointers
	virtual const ArgsT& get_children (void) const = 0;
};

}

#endif // ADE_IFUNCTOR_HPP
