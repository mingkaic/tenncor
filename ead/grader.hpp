///
/// grader.hpp
/// ead
///
/// Purpose:
/// Implement ead gradient definition for supported operations
///

#include <list>

#include "ade/grad_def.hpp"

#include "ead/generated/api.hpp"
#include "ead/generated/grader.hpp"

#include "ead/constant.hpp"

#ifndef EAD_GRADER_HPP
#define EAD_GRADER_HPP

namespace ead
{

template <typename T>
struct GradientBuilder final : public ade::iGradientBuilder
{
	/// Implementation of iGradientBuilder
	ade::TensptrT local_derivative (ade::FuncptrT op,
		size_t arg_idx) const override
	{
		const ade::ArgsT& args = op->get_children();
		ade::TensptrT out;
		switch ((age::_GENERATED_OPCODE) op->get_opcode().code_)
		{
			case age::ABS:
				out = age::div(to_node<T>(args[0].get_tensor()),
					to_node<T>(op))->get_tensor();
				break;
			case age::NEG:
				out = ade::TensptrT(Constant<T>::get_scalar(
					-1, args[0].get_tensor()->shape()));
				break;
			case age::SIN:
				out = age::cos(to_node<T>(args[0].get_tensor()))->get_tensor();
				break;
			case age::COS:
				out = age::neg(age::sin(
					to_node<T>(args[0].get_tensor())))->get_tensor();
				break;
			case age::TAN:
				out = age::div(make_constant_scalar<T>(1,
					args[0].get_tensor()->shape()),
					age::pow(
						age::cos(to_node<T>(args[0].get_tensor())),
						make_constant_scalar<T>(2, args[0].get_tensor()->shape())
					)
				)->get_tensor();
				break;
			case age::EXP:
				out = op;
				break;
			case age::LOG:
				out = age::div(
					make_constant_scalar<T>(1,args[0].get_tensor()->shape()),
					to_node<T>(args[0].get_tensor())
				)->get_tensor();
				break;
			case age::SQRT:
				out = age::div(
					make_constant_scalar<T>(1,args[0].get_tensor()->shape()),
					age::mul(
						make_constant_scalar<T>(2,
							args[0].get_tensor()->shape()),
						to_node<T>(op)
					)
				)->get_tensor();
				break;
			case age::SQUARE:
				out = age::mul(
					make_constant_scalar<T>(2, args[0].get_tensor()->shape()),
					to_node<T>(args[0].get_tensor())
				)->get_tensor();
				break;
			case age::CUBE:
				out = age::mul(
					make_constant_scalar<T>(3, args[0].get_tensor()->shape()),
					age::square(to_node<T>(args[0].get_tensor()))
				)->get_tensor();
				break;
			case age::SIGMOID:
				out = age::sigmoid_grad(
					to_node<T>(args[0].get_tensor()))->get_tensor();
				break;
			case age::SIGMOID_GRAD:
				out = age::mul(
					to_node<T>(op),
					age::sub(
						make_constant_scalar<T>(1,
							args[0].get_tensor()->shape()),
						age::mul(
							make_constant_scalar<T>(2,
								args[0].get_tensor()->shape()),
							age::sigmoid(to_node<T>(args[0].get_tensor()))
						)
					)
				)->get_tensor();
				break;
			case age::TANH:
				out = age::sub(
					make_constant_scalar<T>(1,args[0].get_tensor()->shape()),
					age::square(to_node<T>(op))
				)->get_tensor();
				break;
			case age::ROUND:
			case age::REDUCE_SUM:
			case age::EXTEND:
			case age::PERMUTE:
			case age::ADD:
				out = get_const_one(args[0].get_tensor()->shape());
				break;
			case age::MUL:
			case age::CONV:
				out = args[(size_t)(arg_idx==0)].get_tensor();
				break;
			case age::MAX:
			case age::MIN:
				out = age::eq(to_node<T>(op),
					to_node<T>(args[arg_idx].get_tensor()))->get_tensor();;
				break;
			case age::POW:
				out = (arg_idx==0 ?
					age::mul(
						to_node<T>(args[1].get_tensor()),
						age::pow(
							to_node<T>(args[0].get_tensor()),
							age::sub(
								to_node<T>(args[1].get_tensor()),
								make_constant_scalar<T>(1,
									args[0].get_tensor()->shape())
							)
						)
					) :
					age::mul(age::log(to_node<T>(args[0].get_tensor())),
						to_node<T>(op))
				)->get_tensor();;
				break;
			case age::SUB:
				out = ade::TensptrT(Constant<T>::get_scalar(arg_idx == 0 ?
					1 : -1, args[0].get_tensor()->shape()));
				break;
			case age::DIV:
			{
				auto denom = to_node<T>(args[1].get_tensor());
				out = (arg_idx==0 ?
					age::div(
						make_constant_scalar<T>(1,
							args[0].get_tensor()->shape()),
						denom
					) :
					age::div(
						age::div(
							age::neg(to_node<T>(args[0].get_tensor())),
							denom),
						denom
					))->get_tensor();
			}
				break;
			case age::EQ:
			case age::NEQ:
			case age::GT:
			case age::LT:
			case age::RAND_UNIF:
				out = get_const_zero(args[0].get_tensor()->shape());
				break;
			case age::REDUCE_PROD:
				out = age::div(
					reduce_grad(args[0], to_node<T>(op), arg_idx),
					to_node<T>(args[0].get_tensor())
				)->get_tensor();
				break;
			case age::REDUCE_MAX:
			case age::REDUCE_MIN:
				out = age::eq(
					reduce_grad(args[0], to_node<T>(op), arg_idx),
					to_node<T>(args[0].get_tensor())
				)->get_tensor();
				break;
			case age::MATMUL:
			{
				NodeptrT<T> lhs = to_node<T>(args[0].get_tensor());
				NodeptrT<T> rhs = to_node<T>(args[1].get_tensor());
				out = (0 == arg_idx ?
					// ext_rhs
					age::permute(age::extend(rhs, 2, {
						lhs->shape().at(1)}), {0,2,1}) :
					// ext_lhs
					age::permute(age::extend(lhs, 2, {
						rhs->shape().at(0)}), {2,1,0})
				)->get_tensor();
			}
				break;
			case age::CONV_IMG_GRAD:
				logs::fatal("cannot derive CONV_IMG_GRAD");
				break;
			case age::CONV_KRN_GRAD:
				logs::fatal("cannot derive CONV_KRN_GRAD");
				break;
			default:
				logs::fatal("Unknown op");
		}
		return out;
	}

	/// Implementation of iGradientBuilder
	ade::TensptrT chain_rule (ade::FuncptrT op, const ade::TensptrT& local_der,
		ade::TensptrT supcomp_grad, size_t arg_idx) const override
	{
		NodeptrT<T> out;
		switch (op->get_opcode().code_)
		{
			case age::ABS:
			case age::NEG:
			case age::SIN:
			case age::COS:
			case age::TAN:
			case age::EXP:
			case age::LOG:
			case age::SQRT:
			case age::SQUARE:
			case age::CUBE:
			case age::ROUND:
			case age::SIGMOID:
			case age::SIGMOID_GRAD:
			case age::TANH:
			case age::ADD:
			case age::MUL:
			case age::MAX:
			case age::MIN:
			case age::POW:
			case age::SUB:
			case age::DIV:
			case age::EQ:
			case age::NEQ:
			case age::GT:
			case age::LT:
			case age::RAND_UNIF:
				out = age::mul(to_node<T>(local_der),
					to_node<T>(supcomp_grad));
				break;
			case age::REDUCE_MAX:
			case age::REDUCE_MIN:
			case age::REDUCE_PROD:
			case age::REDUCE_SUM:
				out = age::mul(to_node<T>(local_der), reduce_grad(
					op->get_children()[0], to_node<T>(supcomp_grad), arg_idx));
				break;
			case age::EXTEND:
				out = age::mul(to_node<T>(local_der), extend_grad(
					op.get(), to_node<T>(supcomp_grad), arg_idx));
				break;
			case age::PERMUTE:
				out = age::mul(to_node<T>(local_der), permute_grad(
					op.get(), to_node<T>(supcomp_grad), arg_idx));
				break;
			case age::MATMUL:
				out = age::reduce_sum(
					age::permute(
						age::mul(to_node<T>(local_der),
							age::extend(to_node<T>(supcomp_grad), 2, {
								op->get_children()[0].
									get_tensor()->shape().at(0)
							})),
						0 == arg_idx ?
							std::vector<uint8_t>{2, 1, 0} :
							std::vector<uint8_t>{0, 2, 1}), 2);
				break;
			case age::CONV:
			{
				ade::Opcode opcode;
				auto args = op->get_children();
				ade::CoordptrT fwd_shaper =
					args[(size_t)(0 == arg_idx)].get_shaper();
				ade::CoordptrT rev_shaper(
					args[arg_idx].get_shaper()->reverse());
				if (arg_idx == 0)
				{
					opcode = ade::Opcode{"CONV_IMG_GRAD",
						age::CONV_IMG_GRAD};
				}
				else
				{
					opcode = ade::Opcode{"CONV_KRN_GRAD",
						age::CONV_KRN_GRAD};
				}
				ade::CoordptrT full_shaper(
					fwd_shaper->connect(*rev_shaper));
				out = make_functor<T>(opcode, {
					FuncArg<T>(to_node<T>(local_der), full_shaper, nullptr),
					FuncArg<T>(to_node<T>(supcomp_grad), rev_shaper, nullptr),
				});
			}
				break;
			case age::CONV_IMG_GRAD:
				logs::fatal("cannot derive CONV_IMG_GRAD");
				break;
			case age::CONV_KRN_GRAD:
				logs::fatal("cannot derive CONV_KRN_GRAD");
				break;
			default:
				logs::fatal("Unknown op");
		}
		return out->get_tensor();
	}

	/// Implementation of iGradientBuilder
	ade::TensptrT get_const_one (ade::Shape shape) const override
	{
		return ade::TensptrT(Constant<T>::get_scalar(1, shape));
	}

	/// Implementation of iGradientBuilder
	ade::TensptrT get_const_zero (ade::Shape shape) const override
	{
		return ade::TensptrT(Constant<T>::get_scalar(0, shape));
	}

	/// Implementation of iGradientBuilder
	ade::TensptrT add (ade::TensptrT& lhs, ade::TensptrT& rhs) const override
	{
		return ade::TensptrT(Functor<T>::get(ade::Opcode{"ADD", age::ADD}, {
			identity_map(to_node<T>(lhs)),
			identity_map(to_node<T>(rhs))
		}));
	}
};

/// Derive root with respect to target and optimized
template <typename T>
NodeptrT<T> derive (NodeptrT<T> root, NodeptrT<T> target)
{
	GradientBuilder<T> builder;
	ade::TensptrT derivative = builder.derive(
		root->get_tensor(), target->get_tensor());
	return to_node<T>(derivative);
}

}

#endif // EAD_GRADER_HPP
