/*!
 *
 *  functor.hpp
 *  ade
 *
 *  Purpose:
 *  define traversible tensor extension
 *
 */

#include <algorithm>

#include "util/strify.hpp"

#include "ade/tensor.hpp"
#include "ade/grader.hpp"
#include "ade/fwder.hpp"

#ifndef ADE_FUNCTOR_HPP
#define ADE_FUNCTOR_HPP

namespace ade
{

/*! Traversible extension of Tensor interface */
struct iFunctor : public iTensor
{
	virtual ~iFunctor (void) = default;

	/*! get opcode mapping to forward and gradient operators */
	virtual OPCODE get_code (void) const = 0;

	/*! get arguments as raw pointers */
	virtual std::vector<iTensor*> get_refs (void) const = 0;
};

/*! Tensor implementation representing an operation specified in OPCODE  */
template <OPCODE opcode, typename... Args>
struct Functor final : public iFunctor
{
	static Tensorptr get (std::vector<Tensorptr> args, Args... meta)
	{
		std::tuple<Args...> tp(meta...);
		return new Functor(forwarder<opcode,Args...>(args,
			std::forward<Args>(meta)...), args, tp);
	}

	/*! implementation of iTensor  */
	const Shape& shape (void) const override
	{
		return shape_;
	}

	/*! implementation of iTensor  */
	Tensorptr gradient (Tensorptr& wrt) const override
	{
		if (wrt.get() == this)
		{
			return constant_one(wrt->shape().as_list());
		}
		return grad_helper(wrt, std::index_sequence_for<Args...>());
	}

	/*! implementation of iTensor  */
	std::string to_string (void) const override
	{
		return opname(opcode) + "<" + util::tuple_to_string(meta_) + ">";
	}

	/*! implementation of iFunctor  */
	OPCODE get_code (void) const override
	{
		return opcode;
	}

	/*! implementation of iFunctor  */
	std::vector<iTensor*> get_refs (void) const override
	{
		std::vector<iTensor*> out(args_.size());
		std::transform(args_.begin(), args_.end(), out.begin(),
		[](const Tensorptr& arg)
		{
			return arg.get();
		});
		return out;
	}

	/*! metadata for forward operator  */
	std::tuple<Args...> meta_;

private:
	Functor (Shape shape, std::vector<Tensorptr> args,
		std::tuple<Args...>& meta) :
		meta_(meta), args_(args), shape_(shape) {}

	template <size_t... I>
	Tensorptr grad_helper (Tensorptr& wrt, std::index_sequence<I...>) const
	{
		return grader<opcode,Args...>(args_, wrt, std::get<I>(meta_)...);
	}

	/*! functor argument  */
	std::vector<Tensorptr> args_;

	/*! internal shape  */
	Shape shape_;
};

#define MAPCASE(CODE)case CODE: return Functor<CODE,Args...>::get(args, meta...);

/*! Create functor of opcode determined at runtime  */
template <typename... Args>
Tensorptr runtime_functor (OPCODE opcode,
	std::vector<Tensorptr> args, Args... meta)
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
		MAPCASE(BINO)
		MAPCASE(UNIF)
		MAPCASE(NORM)
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

#endif /* ADE_FUNCTOR_HPP */
