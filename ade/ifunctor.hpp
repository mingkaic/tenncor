///
/// functor.hpp
/// ade
///
/// Purpose:
/// Define functor nodes of an equation graph
///

#include "ade/funcarg.hpp"

#ifndef ADE_IFUNCTOR_HPP
#define ADE_IFUNCTOR_HPP

namespace ade
{

/// Encoding of operation
struct Opcode final
{
	/// String representation of operation
	std::string name_;

	/// Numerical encoding of operation
	size_t code_;
};

/// Interface of iOperation-defined operation node
struct iFunctor : public iTensor
{
	virtual ~iFunctor (void) = default;

	/// Implementation of iTensor
	void accept (iTraveler& visiter) override
	{
		visiter.visit(this);
	}

	/// Return operation encoding
	virtual Opcode get_opcode (void) const = 0;

	/// Return children nodes as a vector of raw pointers
	virtual const ArgsT& get_children (void) const = 0;

	/// Update child at specified index
	virtual void update_child (FuncArg arg, size_t index) = 0;
};

/// Functor smart pointer
using FuncptrT = std::shared_ptr<iFunctor>;

}

#endif // ADE_IFUNCTOR_HPP
