#include <algorithm>

#include "ade/tensor.hpp"
#include "ade/grader.hpp"
#include "ade/fwder.hpp"

#include "util/strify.hpp"

#ifndef ADE_FUNCTOR_HPP
#define ADE_FUNCTOR_HPP

namespace ade
{

// useful for equation graph traversal
struct iFunctor : public iTensor
{
	virtual ~iFunctor (void) = default;

	virtual OPCODE get_code (void) const = 0;

	std::vector<iTensor*> get_refs (void) const
	{
		std::vector<iTensor*> out(args_.size());
		std::transform(args_.begin(), args_.end(), out.begin(),
		[](const Tensorptr& arg)
		{
			return arg.get();
		});
		return out;
	}

protected:
	iFunctor (Shape& shape, std::vector<Tensorptr>& args) :
		iTensor(shape), args_(args) {}

	std::vector<Tensorptr> args_;
};

template <OPCODE opcode, typename... Args>
struct Functor final : public iFunctor
{
	static Tensorptr get (std::vector<Tensorptr> args, Args... meta)
	{
		std::tuple<Args...> tp(meta...);
		return new Functor(forwarder<opcode,Args...>(args,
			std::forward<Args>(meta)...), args, tp);
	}

	Tensorptr gradient (Tensorptr& wrt) const override
	{
		if (wrt.get() == this)
		{
			return constant_one(wrt->shape_.as_list());
		}
		return grad_helper(wrt, std::index_sequence_for<Args...>());
	}

	std::string to_string (void) const override
	{
		return opname(opcode) + "<" + util::tuple_to_string(meta_) + ">";
	}

	OPCODE get_code (void) const override
	{
		return opcode;
	}

private:
	Functor (Shape shape, std::vector<Tensorptr> args,
		std::tuple<Args...>& meta) :
		iFunctor(shape, args), meta_(meta) {}

	template <size_t... I>
	Tensorptr grad_helper (Tensorptr& wrt, std::index_sequence<I...>) const
	{
		return grader<opcode,Args...>(args_, wrt, std::get<I>(meta_)...);
	}

	std::tuple<Args...> meta_;
};

#define MAPCASE(CODE)case CODE: return Functor<CODE,Args...>::get(args, meta...);

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
