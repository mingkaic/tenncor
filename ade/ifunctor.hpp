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

struct iOpcode
{
	virtual ~iOpcode (void) = default;

	virtual std::string to_string (void) const = 0;

	virtual size_t opnum (void) const = 0;

	virtual Tensorptr gradient (ArgsT args, size_t gradidx) const = 0;

	// todo: slowly remove these in favor of better gradient api
	virtual Tensorptr grad_vertical_merge (MappedTensor bot, MappedTensor top) const = 0;

	virtual Tensorptr grad_horizontal_merge (ArgsT& grads) const = 0;
};

using CodePtrT = std::unique_ptr<iOpcode>;

/// Interface of OPCODE-defined operation node
struct iFunctor : public iTensor
{
	virtual ~iFunctor (void) = default;

	/// Implementation of iTensor
	void accept (iTraveler& visiter) override
	{
		visiter.visit(this);
	}

	/// Return OPCODE mapping to forward and gradient operators
	virtual const iOpcode& get_code (void) const = 0;

	/// Return children nodes as a vector of raw pointers
	virtual const ArgsT& get_children (void) const = 0;
};

}

#endif // ADE_IFUNCTOR_HPP
