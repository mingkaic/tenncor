///
///	operation.hpp
///	age
///
///	Purpose:
///	Define operation implementation for ade functors
///

#include "adhoc/age/grader.hpp"

#ifndef AGE_OPERATION_HPP
#define AGE_OPERATION_HPP

namespace age
{

struct Grader final : public ade::iGrader
{
	Grader (const ade::iTensor* target) : ade::iGrader(target) {}

	ade::Tensorptr chain_grad (ade::Tensorptr& wrt_child,
		ade::MappedTensor wrt_me) const override
	{
		return ade::Functor::get(make_code(MUL), {
			{ade::identity, wrt_child},
			{ade::identity, ade::Functor::get(make_code(ADD), {wrt_me})},
		});
	}

	ade::Tensorptr add_grads (ade::ArgsT& grads) const override
	{
		return ade::Functor::get(make_code(ADD), grads);
	}

	ade::Tensorptr get_grad (ade::Opcode opcode, ade::ArgsT args, size_t gradidx) const override
	{
		return gradient((OPCODE) opcode.code_, args, gradidx);
	}

	ade::Tensorptr get_scalar (const ade::Shape& shape, size_t scalar) const override
	{
		if (scalar)
		{
			return shaped_one(shape);
		}
		return shaped_zero(shape);
	}

	void set_scalar (const ade::iTensor* key, size_t scalar) override
	{
		set_grad(key, get_scalar(key->shape(), scalar));
	}

	void set_grad (const ade::iTensor* key, ade::Tensorptr value) override
	{
		grads_.emplace(key, value);
	}

	std::unordered_map<const ade::iTensor*,ade::Tensorptr> grads_;
};

}

#endif // AGE_OPERATION_HPP
