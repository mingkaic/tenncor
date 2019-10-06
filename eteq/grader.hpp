///
/// grader.hpp
/// eteq
///
/// Purpose:
/// Implement eteq gradient definition for supported operations
///

#include <list>

#include "teq/grad_def.hpp"

#include "eteq/generated/api.hpp"

#include "eteq/constant.hpp"

#ifndef ETEQ_GRADER_HPP
#define ETEQ_GRADER_HPP

namespace eteq
{

/// Return reduction operator gradient of reduced functor node (bwd)
template <typename T>
NodeptrT<T> reduce_grad (const teq::FuncArg& child,
	NodeptrT<T> bwd, size_t idx)
{
	const teq::Shape& shape = child.get_tensor()->shape();
	teq::CoordptrT revshaper(child.get_shaper()->reverse());
	CoordptrT revcoord;
	{
		auto coorder = child.get_coorder();
		assert(nullptr != coorder);
		teq::CoordT bcast;
		std::fill(bcast.begin(), bcast.end(), 1);
		coorder->access(
			[&](const teq::MatrixT& args)
			{
				for (teq::RankT i = 0; i < teq::rank_cap; ++i)
				{
					auto d = args[0][i];
					if (d < teq::rank_cap)
					{
						bcast[d] = shape.at(d);
					}
				}
			});
		revcoord = std::make_shared<CoordMap>(bcast, false);
	}
	return make_functor<T>(teq::Opcode{"EXTEND",egen::EXTEND}, {
		FuncArg<T>(bwd, revshaper, revcoord)
	});
}

/// Return permutation gradient of permuted functor node (bwd)
template <typename T>
NodeptrT<T> permute_grad (teq::iFunctor* fwd,
	NodeptrT<T> bwd, size_t idx)
{
	const auto& child = fwd->get_children()[0];
	teq::CoordptrT revshaper(child.get_shaper()->reverse());
	CoordptrT revcoord;
	{
		auto coorder = child.get_coorder();
		assert(nullptr != coorder);
		teq::CoordT order;
		coorder->access(
			[&](const teq::MatrixT& args)
			{
				for (teq::RankT i = 0; i < teq::rank_cap; ++i)
				{
					order[args[0][i]] = i;
				}
			});
		revcoord = std::make_shared<CoordMap>(order, true);
	}
	return make_functor<T>(teq::Opcode{"PERMUTE",egen::PERMUTE},{
		FuncArg<T>(bwd, revshaper, revcoord)
	});
}

/// Return extension gradient of extended functor node (bwd)
template <typename T>
NodeptrT<T> extend_grad (teq::iFunctor* fwd,
	NodeptrT<T> bwd, size_t idx)
{
	const auto& child = fwd->get_children()[0];
	teq::CoordptrT revshaper(child.get_shaper()->reverse());
	CoordptrT revcoord;
	{
		auto coorder = child.get_coorder();
		assert(nullptr != coorder);
		std::vector<teq::RankT> red_dims;
		coorder->access(
			[&](const teq::MatrixT& args)
			{
				for (teq::RankT i = 0; i < teq::rank_cap; ++i)
				{
					if (args[0][i] > 1)
					{
						red_dims.push_back(i);
					}
				}
			});
		revcoord = reduce(red_dims);
	}
	return make_functor<T>(teq::Opcode{"REDUCE_SUM",egen::REDUCE_SUM},{
		FuncArg<T>(bwd, revshaper, revcoord)
	});
}

/// ETEQ implementation of TEQ's Backward Propagation Builder
template <typename T>
struct GradientBuilder final : public teq::iGradientBuilder
{
	/// Implementation of iGradientBuilder
	teq::TensptrT local_derivative (teq::FuncptrT op,
		size_t arg_idx) const override
	{
		const teq::ArgsT& args = op->get_children();
		NodeptrT<T> out = nullptr;
		teq::Opcode opcode = op->get_opcode();
		switch ((egen::_GENERATED_OPCODE) opcode.code_)
		{
			case egen::ABS:
				out = TO_NODE(args[0].get_tensor()) / TO_NODE(op);
				break;
			case egen::NEG:
				out = make_constant_scalar<T>(
					-1, args[0].get_tensor()->shape());
				break;
			case egen::SIN:
				out = tenncor::cos(TO_NODE(args[0].get_tensor()));
				break;
			case egen::COS:
				out = -tenncor::sin(TO_NODE(args[0].get_tensor()));
				break;
			case egen::TAN:
				out = (T) 1 / tenncor::pow(
					tenncor::cos(TO_NODE(args[0].get_tensor())), (T) 2);
				break;
			case egen::EXP:
				out = TO_NODE(op);
				break;
			case egen::LOG:
				out = (T) 1 / TO_NODE(args[0].get_tensor());
				break;
			case egen::SQRT:
				out = (T) 1 / ((T) 2 * TO_NODE(op));
				break;
			case egen::SQUARE:
				out = (T) 2 * TO_NODE(args[0].get_tensor());
				break;
			case egen::CUBE:
				out = (T) 3 * tenncor::square(TO_NODE(args[0].get_tensor()));
				break;
			case egen::SIGMOID:
				out = tenncor::sigmoid_grad(
					TO_NODE(args[0].get_tensor()));
				break;
			case egen::SIGMOID_GRAD:
				out = TO_NODE(op) * ((T) 1 - (T) 2 *
					tenncor::sigmoid(TO_NODE(args[0].get_tensor())));
				break;
			case egen::TANH:
				out = (T) 1 - tenncor::square(TO_NODE(op));
				break;
			case egen::ROUND:
			case egen::REDUCE_SUM:
			case egen::EXTEND:
			case egen::PERMUTE:
			case egen::ADD:
			case egen::SLICE:
			case egen::PAD:
			case egen::STRIDE: // todo: figure out if this belongs here
				out = make_constant_scalar<T>(1, args[0].get_tensor()->shape());
				break;
			case egen::MUL:
			case egen::CONV:
				out = TO_NODE(args[(size_t)(arg_idx==0)].get_tensor());
				break;
			case egen::MAX:
			case egen::MIN:
				out = TO_NODE(op) == TO_NODE(args[arg_idx].get_tensor());
				break;
			case egen::POW:
				out = arg_idx==0 ?
					TO_NODE(args[1].get_tensor()) *
					tenncor::pow(
						TO_NODE(args[0].get_tensor()),
						TO_NODE(args[1].get_tensor()) - (T) 1
					) :
					tenncor::log(TO_NODE(args[0].get_tensor())) *
						TO_NODE(op);
				break;
			case egen::SUB:
				out = make_constant_scalar<T>(arg_idx == 0 ?
					1 : -1, args[0].get_tensor()->shape());
				break;
			case egen::DIV:
			{
				auto denom = TO_NODE(args[1].get_tensor());
				out = arg_idx==0 ?
					(T) 1 / denom :
					-TO_NODE(args[0].get_tensor()) / denom / denom;
			}
				break;
			case egen::EQ:
			case egen::NEQ:
			case egen::GT:
			case egen::LT:
			case egen::RAND_UNIF:
			case egen::SELECT:
				out = make_constant_scalar<T>(0, args[0].get_tensor()->shape());
				break;
			case egen::REDUCE_PROD: // todo: prevent divide by zero
				out =
					reduce_grad(args[0], TO_NODE(op), arg_idx) /
					TO_NODE(args[0].get_tensor());
				break;
			case egen::REDUCE_MAX:
			case egen::REDUCE_MIN:
				out =
					reduce_grad(args[0], TO_NODE(op), arg_idx) ==
					TO_NODE(args[0].get_tensor());
				break;
			case egen::MATMUL:
			{
				NodeptrT<T> lhs = TO_NODE(args[0].get_tensor());
				NodeptrT<T> rhs = TO_NODE(args[1].get_tensor());
				out = 0 == arg_idx ?
					// ext_rhs
					tenncor::permute(tenncor::extend(rhs, 2, {
						lhs->shape().at(1)}), {0,2,1}) :
					// ext_lhs
					tenncor::permute(tenncor::extend(lhs, 2, {
						rhs->shape().at(0)}), {2,1,0});
			}
				break;
			case egen::CONV_IMG_GRAD:
				logs::fatal("cannot derive CONV_IMG_GRAD");
				break;
			case egen::CONV_KRN_GRAD:
				logs::fatal("cannot derive CONV_KRN_GRAD");
				break;
			default:
				logs::fatalf("Unknown op %s", opcode.name_.c_str());
		}
		return out->get_tensor();
	}

	/// Implementation of iGradientBuilder
	teq::TensptrT chain_rule (teq::FuncptrT op, const teq::TensptrT& local_der,
		teq::TensptrT supcomp_grad, size_t arg_idx) const override
	{
		NodeptrT<T> out = nullptr;
		teq::Opcode opcode = op->get_opcode();
		switch (opcode.code_)
		{
			case egen::ABS:
			case egen::NEG:
			case egen::SIN:
			case egen::COS:
			case egen::TAN:
			case egen::EXP:
			case egen::LOG:
			case egen::SQRT:
			case egen::SQUARE:
			case egen::CUBE:
			case egen::ROUND:
			case egen::SIGMOID:
			case egen::SIGMOID_GRAD:
			case egen::TANH:
			case egen::ADD:
			case egen::MUL:
			case egen::MAX:
			case egen::MIN:
			case egen::POW:
			case egen::SUB:
			case egen::DIV:
			case egen::EQ:
			case egen::NEQ:
			case egen::GT:
			case egen::LT:
			case egen::RAND_UNIF:
				out = TO_NODE(local_der) *
					TO_NODE(supcomp_grad);
				break;
			case egen::REDUCE_MAX:
			case egen::REDUCE_MIN:
			case egen::REDUCE_PROD:
			case egen::REDUCE_SUM:
				out = TO_NODE(local_der) * reduce_grad(
					op->get_children()[0], TO_NODE(supcomp_grad), arg_idx);
				break;
			case egen::EXTEND:
				out = TO_NODE(local_der) * extend_grad(
					op.get(), TO_NODE(supcomp_grad), arg_idx);
				break;
			case egen::PERMUTE:
				out = TO_NODE(local_der) * permute_grad(
					op.get(), TO_NODE(supcomp_grad), arg_idx);
				break;
			case egen::MATMUL:
				out = tenncor::reduce_sum(
					tenncor::permute(
						TO_NODE(local_der) *
							tenncor::extend(TO_NODE(supcomp_grad), 2, {
								op->get_children()[0].
									get_tensor()->shape().at(0)
							}),
						0 == arg_idx ?
							std::vector<teq::RankT>{2, 1, 0} :
							std::vector<teq::RankT>{0, 2, 1}), 2, 1);
				break;
			case egen::CONV:
			{
				teq::Opcode opcode;
				auto args = op->get_children();
				teq::CoordptrT fwd_shaper =
					args[(size_t)(0 == arg_idx)].get_shaper();
				teq::CoordptrT rev_shaper(
					args[arg_idx].get_shaper()->reverse());
				if (arg_idx == 0)
				{
					opcode = teq::Opcode{"CONV_IMG_GRAD",
						egen::CONV_IMG_GRAD};
				}
				else
				{
					opcode = teq::Opcode{"CONV_KRN_GRAD",
						egen::CONV_KRN_GRAD};
				}
				teq::CoordptrT full_shaper(
					fwd_shaper->connect(*rev_shaper));
				out = make_functor<T>(opcode, {
					FuncArg<T>(TO_NODE(local_der), full_shaper, nullptr),
					FuncArg<T>(TO_NODE(supcomp_grad), rev_shaper, nullptr),
				});
			}
				break;
			case egen::SLICE:
			{
				auto& child = op->get_children()[0];
				teq::ShapeT offsets;
				teq::ShapeT extents;
				child.get_coorder()->access(
					[&](const teq::MatrixT& args)
					{
						std::copy(args[0], args[0] + teq::rank_cap, offsets.begin());
						std::copy(args[1], args[1] + teq::rank_cap, extents.begin());
					});
				teq::Shape cshape = child.get_tensor()->shape();
				eteq::PairVecT<teq::DimT> paddings;
				paddings.reserve(teq::rank_cap);
				for (size_t i = 0; i < teq::rank_cap; ++i)
				{
					teq::DimT leftpad = offsets[i];
					paddings.push_back({leftpad,
						cshape.at(i) - (leftpad + extents[i])});
				}
				out = TO_NODE(local_der) *
					tenncor::pad(TO_NODE(supcomp_grad), paddings);
			}
				break;
			case egen::PAD:
			{
				teq::CoordT paddings;
				auto& child = op->get_children()[0];
				teq::ShapeT leftpad;
				teq::ShapeT rightpad;
				child.get_coorder()->access(
					[&](const teq::MatrixT& args)
					{
						std::copy(args[0], args[0] + teq::rank_cap, leftpad.begin());
						std::copy(args[1], args[1] + teq::rank_cap, rightpad.begin());
					});
				teq::Shape oshape = op->shape();
				eteq::PairVecT<teq::DimT> extents;
				extents.reserve(teq::rank_cap);
				for (size_t i = 0; i < teq::rank_cap; ++i)
				{
					teq::DimT offset = leftpad[i];
					extents.push_back({offset,
						oshape.at(i) - rightpad[i] - offset});
				}
				out = TO_NODE(local_der) *
					tenncor::slice(TO_NODE(supcomp_grad), extents);
			}
				break;
			case egen::SELECT:
			{
				if (0 == arg_idx)
				{
					out = TO_NODE(local_der);
					break;
				}
				auto condition = TO_NODE(
					op->get_children()[0].get_tensor());
				auto then = TO_NODE(supcomp_grad);
				auto otherwise = make_constant_scalar<T>(0, op->shape());
				if (1 < arg_idx)
				{
					std::swap(then, otherwise);
				}
				out = tenncor::if_then_else(condition, then, otherwise);
			}
				break;
			case egen::CONV_IMG_GRAD:
				logs::fatal("cannot derive CONV_IMG_GRAD");
				break;
			case egen::CONV_KRN_GRAD:
				logs::fatal("cannot derive CONV_KRN_GRAD");
				break;
			case egen::STRIDE: // todo: implement
			default:
				logs::fatalf("Unknown op %s", opcode.name_.c_str());
		}
		return out->get_tensor();
	}

	/// Implementation of iGradientBuilder
	teq::TensptrT get_const_one (teq::Shape shape) const override
	{
		return make_constant_scalar<T>(1, shape)->get_tensor();
	}

	/// Implementation of iGradientBuilder
	teq::TensptrT get_const_zero (teq::Shape shape) const override
	{
		return make_constant_scalar<T>(0, shape)->get_tensor();
	}

	/// Implementation of iGradientBuilder
	teq::TensptrT add (teq::TensptrT& lhs, teq::TensptrT& rhs) const override
	{
		return teq::TensptrT(Functor<T>::get(teq::Opcode{"ADD", egen::ADD}, {
			identity_map(TO_NODE(lhs)),
			identity_map(TO_NODE(rhs))
		}));
	}
};

/// Derive root with respect to target and optimized
template <typename T>
NodeptrT<T> derive (NodeptrT<T> root, NodeptrT<T> target)
{
	GradientBuilder<T> builder;
	teq::TensptrT derivative = builder.derive(
		root->get_tensor(), target->get_tensor());
	return TO_NODE(derivative);
}

}

#endif // ETEQ_GRADER_HPP
