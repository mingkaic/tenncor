#include <algorithm>

#include "ade/tensor.hpp"
#include "ade/grader.hpp"
#include "ade/fwder.hpp"

#include "util/strify.hpp"

#ifndef ADE_FUNCTOR_HPP
#define ADE_FUNCTOR_HPP

namespace ade
{

template <OPCODE opcode, typename... Args>
struct Functor final : public iTensor
{
	static Tensorptr get (std::vector<Tensorptr> args, Args... meta)
	{
		std::tuple<Args...> tp(meta...);
		return new Functor(
			forwarder<opcode,Args...>(args, std::forward<Args>(meta)...), args, tp);
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

private:
	Functor (Shape shape, std::vector<Tensorptr> args,
		std::tuple<Args...>& meta) : iTensor(shape), args_(args), meta_(meta) {}

	template <size_t... I>
	Tensorptr grad_helper (Tensorptr& wrt, std::index_sequence<I...>) const
	{
		return grader<opcode,Args...>(args_, wrt, std::get<I>(meta_)...);
	}

	std::vector<Tensorptr> args_;

	std::tuple<Args...> meta_;
};

}

#endif /* ADE_FUNCTOR_HPP */
