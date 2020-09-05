///
/// functor.hpp
/// teq
///
/// Purpose:
/// Define functor nodes of an equation graph
///

#ifndef TEQ_IFUNCTOR_HPP
#define TEQ_IFUNCTOR_HPP

#include "internal/teq/objs.hpp"

namespace teq
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
struct iFunctor : public iTensor, public marsh::iAttributed
{
	virtual ~iFunctor (void) = default;

	iFunctor* clone (void) const
	{
		return static_cast<iFunctor*>(this->clone_impl());
	}

	/// Implementation of iTensor
	void accept (iTraveler& visiter) override
	{
		visiter.visit(*this);
	}

	/// Return operation encoding
	virtual Opcode get_opcode (void) const = 0;

	/// Return vector of functor arguments
	virtual TensptrsT get_args (void) const = 0;

	/// Update child at specified index
	virtual void update_child (TensptrT arg, size_t index) = 0;
};

using FuncptrT = std::shared_ptr<iFunctor>;

using FuncsT = std::vector<iFunctor*>;

using FuncSetT = std::unordered_set<iFunctor*>;

template <typename T>
using FuncMapT = std::unordered_map<iFunctor*,T>;

}

#endif // TEQ_IFUNCTOR_HPP
