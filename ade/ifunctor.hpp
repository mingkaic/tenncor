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

#include "ade/opcode.hpp"
#include "ade/tensor.hpp"

#ifndef ADE_IFUNCTOR_HPP
#define ADE_IFUNCTOR_HPP

namespace ade
{

/// Type of functor arguments
using ArgsT = std::vector<MappedTensor>;

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
	virtual OPCODE get_code (void) const = 0;

	/// Return children nodes as a vector of raw pointers
	virtual const ArgsT& get_children (void) const = 0;
};

}

#endif // ADE_IFUNCTOR_HPP
