///
///	functor.hpp
///	ade
///
///	Purpose:
///	Define functor nodes of an equation graph
///

#include <algorithm>

#include "ade/string.hpp"

#include "ade/tensor.hpp"
#include "ade/grader.hpp"
#include "ade/fwder.hpp"

#ifndef ADE_FUNCTOR_HPP
#define ADE_FUNCTOR_HPP

namespace ade
{

/// Interface of OPCODE-defined operation node
struct iFunctor : public iTensor
{
	virtual ~iFunctor (void) = default;

	/// Implementation of iTensor
	void accept (Traveler& visiter) override
	{
		visiter.visit(this);
	}

	/// Return OPCODE mapping to forward and gradient operators
	virtual OPCODE get_code (void) const = 0;

	/// Return children nodes as a vector of raw pointers
	virtual std::vector<iTensor*> get_children (void) const = 0;
};

/// Functor of the graph mapping to operators specified in OPCODE template
/// ARGS template captures non-tensor arguments used for certain operators
template <OPCODE OP, typename... ARGS>
struct Functor final : public iFunctor
{
	/// Return a Functor with with input tensor and meta arguments
	static Functor<OP,ARGS...>* get (std::vector<Tensorptr> args, ARGS... meta)
	{
		std::tuple<ARGS...> tp(meta...);
		return new Functor(forwarder<OP,ARGS...>(args,
			std::forward<ARGS>(meta)...), args, tp);
	}

	/// Implementation of iTensor
	const Shape& shape (void) const override
	{
		return shape_;
	}

	/// Implementation of iTensor
	Tensorptr gradient (Tensorptr& wrt) const override
	{
		if (wrt.get() == this)
		{
			return constant_one(wrt->shape());
		}
		return grad_helper(wrt, std::index_sequence_for<ARGS...>());
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return opname(OP) + "<" + ade::to_string(meta_) + ">";
	}

	/// Implementation of iFunctor
	OPCODE get_code (void) const override
	{
		return OP;
	}

	/// Implementation of iFunctor
	std::vector<iTensor*> get_children (void) const override
	{
		std::vector<iTensor*> out(args_.size());
		std::transform(args_.begin(), args_.end(), out.begin(),
		[](const Tensorptr& arg)
		{
			return arg.get();
		});
		return out;
	}

	/// Return extra non-tensor arguments
	const std::tuple<ARGS...>& meta (void) const
	{
		return meta_;
	}

private:
	Functor (Shape shape, std::vector<Tensorptr> args,
		std::tuple<ARGS...>& meta) :
		args_(args), meta_(meta), shape_(shape) {}

	template <size_t... I>
	Tensorptr grad_helper (Tensorptr& wrt, std::index_sequence<I...>) const
	{
		return grader<OP,ARGS...>(args_, wrt, std::get<I>(meta_)...);
	}

	/// Tensor arguments (and children)
	std::vector<Tensorptr> args_;

	/// Extra arguments for certain operators
	/// These arguments are hidden to ensure shape is correct
	/// since meta data can influence shape
	std::tuple<ARGS...> meta_;

	/// Shape info built at construction time according to arguments
	Shape shape_;
};

#define MAPCASE(CODE)case CODE:\
return Functor<CODE,ARGS...>::get(args, meta...);

/// Return Functor of non-template OPCODE useful for runtime OPCODE generation
template <typename... ARGS>
Tensorptr runtime_functor (OPCODE opcode,
	std::vector<Tensorptr> args, ARGS... meta)
{
	switch (opcode)
	{
		MAPCASE(ABS)
		MAPCASE(NEG)
		MAPCASE(NOT)
		MAPCASE(SIN)
		MAPCASE(COS)
		MAPCASE(TAN)
		MAPCASE(EXP)
		MAPCASE(LOG)
		MAPCASE(SQRT)
		MAPCASE(ROUND)
		MAPCASE(FLIP)
		MAPCASE(POW)
		MAPCASE(ADD)
		MAPCASE(SUB)
		MAPCASE(MUL)
		MAPCASE(DIV)
		MAPCASE(EQ)
		MAPCASE(NE)
		MAPCASE(LT)
		MAPCASE(GT)
		MAPCASE(MIN)
		MAPCASE(MAX)
		MAPCASE(RAND_BINO)
		MAPCASE(RAND_UNIF)
		MAPCASE(RAND_NORM)
		MAPCASE(N_ELEMS)
		MAPCASE(N_DIMS)
		MAPCASE(ARGMAX)
		MAPCASE(RMAX)
		MAPCASE(RSUM)
		MAPCASE(MATMUL)
		MAPCASE(PERMUTE)
		MAPCASE(EXTEND)
		MAPCASE(RESHAPE)
		default:
			throw std::bad_function_call();
	}
}

#undef MAPCASE

}

#endif // ADE_FUNCTOR_HPP
