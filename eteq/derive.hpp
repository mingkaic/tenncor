///
/// derive.hpp
/// eteq
///
/// Purpose:
/// Implement eteq gradient definition for supported operations
///

#include "teq/derive.hpp"

#include "eigen/operator.hpp"

#include "eteq/generated/api.hpp"
#include "eteq/constant.hpp"

#ifndef ETEQ_DERIVE_HPP
#define ETEQ_DERIVE_HPP

namespace eteq
{

/// Return reduction operator gradient of reduced functor node (bwd)
template <typename T>
ETensor<T> reduce_grad (teq::Shape shape,
	ETensor<T> bwd, teq::FuncptrT fwd)
{
	std::vector<teq::DimT> bcast(teq::rank_cap, 1);

	std::set<teq::RankT> ranks;
	eigen::Packer<std::set<teq::RankT>>().unpack(ranks, *fwd);

	for (teq::RankT d : ranks)
	{
		if (d < teq::rank_cap)
		{
			bcast[d] = shape.at(d);
		}
	}
	return tenncor::extend(bwd, bcast);
}

static inline std::vector<teq::RankT> reorder_permute (
	std::vector<teq::RankT> order)
{
	std::array<bool,teq::rank_cap> visited;
	std::fill(visited.begin(), visited.end(), false);
	for (teq::RankT i = 0, n = order.size(); i < n; ++i)
	{
		visited[order[i]] = true;
	}
	for (teq::RankT i = 0; i < teq::rank_cap; ++i)
	{
		if (false == visited[i])
		{
			order.push_back(i);
		}
	}
	std::vector<teq::RankT> reorder(teq::rank_cap);
	for (size_t i = 0; i < teq::rank_cap; ++i)
	{
		reorder[order[i]] = i;
	}
	return reorder;
}

/// ETEQ implementation of TEQ's Backward Propagation Builder
template <typename T>
struct DerivativeFuncs final : public teq::iDerivativeFuncs
{
	/// Implementation of iDerivativeFuncs
	teq::TensptrT local_derivative (teq::FuncptrT op,
		size_t arg_idx) const override
	{
		auto args = op->get_children();
		ETensor<T> out;
		teq::Opcode opcode = op->get_opcode();
		switch ((egen::_GENERATED_OPCODE) opcode.code_)
		{
			case egen::ABS:
				out = ETensor<T>(args.front()) / ETensor<T>(op);
				break;
			case egen::NEG:
				out = make_constant_scalar<T>(
					-1, args.front()->shape());
				break;
			case egen::SIN:
				out = tenncor::cos(ETensor<T>(args.front()));
				break;
			case egen::COS:
				out = -tenncor::sin(ETensor<T>(args.front()));
				break;
			case egen::TAN:
				out = (T) 1 / tenncor::pow(
					tenncor::cos(ETensor<T>(args.front())), (T) 2);
				break;
			case egen::EXP:
				out = ETensor<T>(op);
				break;
			case egen::LOG:
				out = (T) 1 / ETensor<T>(args.front());
				break;
			case egen::SQRT:
				out = (T) 1 / ((T) 2 * ETensor<T>(op));
				break;
			case egen::SQUARE:
				out = (T) 2 * ETensor<T>(args.front());
				break;
			case egen::CUBE:
				out = (T) 3 * tenncor::square(ETensor<T>(args.front()));
				break;
			case egen::SIGMOID:
				out = ETensor<T>(op) * ((T) 1 - ETensor<T>(op));
				break;
			case egen::TANH:
				out = (T) 1 - tenncor::square(ETensor<T>(op));
				break;
			case egen::ROUND:
			case egen::REDUCE_SUM:
			case egen::EXTEND:
			case egen::PERMUTE:
			case egen::RESHAPE:
			case egen::ADD:
			case egen::GROUP_SUM:
			case egen::SLICE:
			case egen::PAD:
			case egen::STRIDE:
			case egen::SCATTER:
			case egen::REVERSE:
			case egen::CONV:
			case egen::CONCAT:
			case egen::GROUP_CONCAT:
			case egen::MATMUL:
				out = make_constant_scalar<T>(1, args[arg_idx]->shape());
				break;
			case egen::MUL:
			case egen::GROUP_PROD:
			{
				ETensorsT<T> nodes;
				size_t nargs = args.size();
				nodes.reserve(nargs);
				for (size_t i = 0, n = nargs; i < n; ++i)
				{
					if (i != arg_idx)
					{
						nodes.push_back(ETensor<T>(args[i]));
					}
				}
				out = tenncor::prod(nodes);
			}
				break;
			case egen::MAX:
			case egen::MIN:
				out = ETensor<T>(op) == ETensor<T>(args[arg_idx]);
				break;
			case egen::POW:
				out = arg_idx==0 ?
					ETensor<T>(args[1]) *
					tenncor::pow(
						ETensor<T>(args[0]),
						ETensor<T>(args[1]) - (T) 1
					) :
					tenncor::log(ETensor<T>(args[0])) *
						ETensor<T>(op);
				break;
			case egen::SUB:
				out = make_constant_scalar<T>(arg_idx == 0 ?
					1 : -1, args[0]->shape());
				break;
			case egen::DIV:
			{
				auto denom = ETensor<T>(args[1]);
				out = arg_idx==0 ?
					(T) 1 / denom :
					-ETensor<T>(args[0]) / denom / denom;
			}
				break;
			case egen::EQ:
			case egen::NEQ:
			case egen::GT:
			case egen::LT:
			case egen::RAND_UNIF:
			case egen::SELECT:
				out = make_constant_scalar<T>(0, args.front()->shape());
				break;
			case egen::REDUCE_PROD: // todo: prevent divide by zero
				out =
					reduce_grad(args.front()->shape(), ETensor<T>(op), op) /
					ETensor<T>(args.front());
				break;
			case egen::REDUCE_MAX:
			case egen::REDUCE_MIN:
				out = reduce_grad(args.front()->shape(), ETensor<T>(op), op) == ETensor<T>(args.front());
				break;
			case egen::ARGMAX:
				teq::fatalf("cannot derive %s", opcode.name_.c_str());
				break;
			default:
				teq::fatalf("Unknown op %s", opcode.name_.c_str());
		}
		return (teq::TensptrT) out;
	}

	/// Implementation of iDerivativeFuncs
	teq::TensptrT chain_rule (teq::FuncptrT op, const teq::TensptrT& local_der,
		teq::TensptrT supcomp_grad, size_t arg_idx) const override
	{
		ETensor<T> out;
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
			case egen::TANH:
			case egen::ADD:
			case egen::GROUP_SUM:
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
				out = ETensor<T>(local_der) * ETensor<T>(supcomp_grad);
				break;
			case egen::REDUCE_MAX:
			case egen::REDUCE_MIN:
			case egen::REDUCE_PROD:
			case egen::REDUCE_SUM:
				out = ETensor<T>(local_der) * reduce_grad(
					op->get_children().front()->shape(),
					ETensor<T>(supcomp_grad), op);
				break;
			case egen::EXTEND:
			{
				std::vector<teq::DimT> bcast = eigen::unpack_extend(
					op->get_children().front()->shape(), *op);

				std::set<teq::RankT> dims;
				// technically, reduce_sum is not grad of broadcast,
				// (since broadcast works on dimension > 1) (todo: account for this)
				// but assuming broadcast is applied on dimensions of 1, reduce_sum is sufficient
				for (size_t i = 0, n = std::min((size_t) teq::rank_cap, bcast.size());
					i < n; ++i)
				{
					teq::DimT d = bcast[i];
					if (d > 1)
					{
						dims.emplace(i);
					}
				}
				out = ETensor<T>(local_der) * tenncor::reduce_sum(ETensor<T>(supcomp_grad), dims);
			}
				break;
			case egen::PERMUTE:
			{
				std::vector<teq::RankT> order;
				eigen::Packer<std::vector<teq::RankT>>().unpack(order, *op);

				out = ETensor<T>(local_der) * tenncor::permute(
					ETensor<T>(supcomp_grad), reorder_permute(order));
			}
				break;
			case egen::RESHAPE:
			{
				out = ETensor<T>(local_der) * tenncor::reshape(
					ETensor<T>(supcomp_grad),
					op->get_children().front()->shape());
			}
				break;
			case egen::MATMUL:
			{
				auto args = op->get_children();
				eigen::PairVecT<teq::RankT> dims;
				eigen::Packer<eigen::PairVecT<teq::RankT>>().unpack(dims, *op);

				// Given dimension u = dims,

				// ext(dC, Vb[u])[Vb[^u], Va[^u], Vb[u]] @ ext(B, Va[u])[Vb, Va[u]] =>
				// _A[Va[u], Va[^u]] => permute(_A) => dA[Va]

				// ext(A, Vb[u])[Va, Vb[u]] @ ext(dC, Va[u])[Vb[^u], Va[^u], Va[u]] =>
				// _B[Vb[^u], Vb[u]] => permute(_B) => dB[Vb]

				std::array<bool,teq::rank_cap> lvisit;
				std::array<bool,teq::rank_cap> rvisit;
				std::fill(lvisit.begin(), lvisit.end(), false);
				std::fill(rvisit.begin(), rvisit.end(), false);
				// supcomp has <rucom_ranks, lucom_ranks>
				// where rucom_ranks are visited right ranks
				// and lucom_ranks are visited left ranks
				std::vector<teq::RankT>
				lucom_ranks, rucom_ranks, lcom_ranks, rcom_ranks;
				lcom_ranks.reserve(dims.size());
				rcom_ranks.reserve(dims.size());
				lucom_ranks.reserve(teq::rank_cap - dims.size());
				rucom_ranks.reserve(teq::rank_cap - dims.size());
				for (auto coms : dims)
				{
					lvisit[coms.first] = true;
					rvisit[coms.second] = true;
					lcom_ranks.push_back(coms.first);
					rcom_ranks.push_back(coms.second);
				}
				for (teq::RankT i = 0,
					n = teq::narrow_shape(args[0]->shape()).size(); i < n; ++i)
				{
					if (false == lvisit[i])
					{
						lucom_ranks.push_back(i);
					}
				}
				for (teq::RankT i = 0,
					n = teq::narrow_shape(args[1]->shape()).size(); i < n; ++i)
				{
					if (false == rvisit[i])
					{
						rucom_ranks.push_back(i);
					}
				}

				// left = supcomp_grad
				eteq::ETensor<T> right;
				std::vector<teq::RankT> order;
				eigen::PairVecT<teq::RankT> grad_dims;
				if (arg_idx == 0)
				{
					right = args[1];
					for (teq::RankT i = 0, n = rucom_ranks.size(); i < n; ++i)
					{
						grad_dims.push_back({i, rucom_ranks[i]});
					}
					// convolution output has shape <lucom, lcom>
					order = lcom_ranks;
					order.insert(order.end(), lucom_ranks.begin(), lucom_ranks.end());
					// reverse order such that convolution output permutes to args[0]->shape()
					order = reorder_permute(order);
				}
				else
				{
					right = args[0];
					for (teq::RankT i = 0, n = lucom_ranks.size(); i < n; ++i)
					{
						grad_dims.push_back({rucom_ranks.size() + i, lucom_ranks[i]});
					}
					// convolution output has shape <rcom, rucom>
					order = rcom_ranks;
					order.insert(order.end(), rucom_ranks.begin(), rucom_ranks.end());
					// reverse order such that convolution output permutes to args[1]->shape()
					order = reorder_permute(order);
				}
				if (grad_dims.empty())
				{
					grad_dims.push_back({
						teq::narrow_shape(supcomp_grad->shape()).size(),
						teq::narrow_shape(right->shape()).size()});
				}
				out = tenncor::permute(tenncor::contract(
					eteq::ETensor<T>(supcomp_grad), right, grad_dims), order);
			}
				break;
			case egen::CONV:
			{
				// for convolution(X, Y) = C
				auto args = op->get_children();

				std::vector<teq::RankT> order;
				eigen::Packer<std::vector<teq::RankT>>().unpack(order, *op);

				std::vector<teq::RankT> dims;
				for (size_t i = 0, n = std::min((size_t) teq::rank_cap, order.size());
					i < n && order[i] < teq::rank_cap; ++i)
				{
					dims.push_back(order[i]);
				}
				if (arg_idx == 0)
				{
					// convolve(pad(C_grad_sup, Y.shape[dims]-1), reverse(Y))
					size_t ndims = dims.size();
					teq::Shape kernshape = args[1]->shape();
					eigen::PairVecT<teq::DimT> paddings(teq::rank_cap, {0, 0});
					for (size_t i = 0; i < ndims; ++i)
					{
						teq::DimT kpad = kernshape.at(i) - 1;
						paddings[dims[i]] = {kpad, kpad};
					}
					std::vector<teq::RankT> revdims(ndims);
					std::iota(revdims.begin(), revdims.end(), 0);
					out = tenncor::convolution(tenncor::pad(
						ETensor<T>(supcomp_grad), paddings),
						tenncor::reverse(
							ETensor<T>(args[1]),
							std::set<teq::RankT>(revdims.begin(), revdims.end())), dims);
				}
				else
				{
					// convolve(X, C_grad_sup)
					std::vector<teq::RankT> indices(teq::rank_cap);
					std::iota(indices.begin(), indices.end(), 0);
					out = tenncor::permute(
						tenncor::convolution(
							ETensor<T>(args[0]),
							ETensor<T>(supcomp_grad),
							indices), dims);
				}
			}
				break;
			case egen::SLICE:
			{
				eigen::PairVecT<teq::DimT> extents;
				eigen::Packer<eigen::PairVecT<teq::DimT>>().unpack(extents, *op);

				teq::Shape cshape = op->get_children().front()->shape();
				eigen::PairVecT<teq::DimT> paddings;
				paddings.reserve(teq::rank_cap);
				for (size_t i = 0, n = std::min(extents.size(),
					(size_t) teq::rank_cap); i < n; ++i)
				{
					teq::DimT offset = std::min(extents[i].first,
						(teq::DimT) (cshape.at(i) - 1));
					teq::DimT extent = std::min(extents[i].second,
						(teq::DimT) (cshape.at(i) - offset));
					paddings.push_back({offset, cshape.at(i) - (offset + extent)});
				}
				out = ETensor<T>(local_der) *
					tenncor::pad(ETensor<T>(supcomp_grad), paddings);
			}
				break;
			case egen::PAD:
			{
				eigen::PairVecT<teq::DimT> paddings;
				eigen::Packer<eigen::PairVecT<teq::DimT>>().unpack(paddings, *op);

				teq::Shape oshape = op->shape();
				eigen::PairVecT<teq::DimT> extents;
				extents.reserve(teq::rank_cap);
				for (size_t i = 0; i < std::min(paddings.size(),
					(size_t) teq::rank_cap); ++i)
				{
					teq::DimT offset = paddings[i].first;
					extents.push_back({offset,
						oshape.at(i) - paddings[i].second - offset});
				}
				out = ETensor<T>(local_der) *
					tenncor::slice(ETensor<T>(supcomp_grad), extents);
			}
				break;
			case egen::CONCAT:
			{
				auto children = op->get_children();
				teq::Shape cshape = children[arg_idx]->shape();
				teq::RankT axis;
				eigen::Packer<teq::RankT>().unpack(axis, *op);
				teq::DimT offset = 0;
				teq::DimT extent = cshape.at(axis);
				if (arg_idx)
				{
					teq::Shape first_shape = children[0]->shape();

					offset = first_shape.at(axis);
				}
				out = ETensor<T>(local_der) *
					tenncor::slice(ETensor<T>(supcomp_grad), offset, extent, axis);
			}
				break;
			case egen::GROUP_CONCAT: // todo: combine concat and group_concat
			{
				teq::RankT axis;
				eigen::Packer<teq::RankT>().unpack(axis, *op);
				out = ETensor<T>(local_der) *
					tenncor::slice(ETensor<T>(supcomp_grad), arg_idx, 1, axis);
			}
				break;
			case egen::STRIDE:
			{
				std::vector<teq::DimT> incrs;
				eigen::Packer<std::vector<teq::DimT>>().unpack(incrs, *op);

				teq::Shape origshape = op->get_children()[0]->shape();
				out = ETensor<T>(local_der) * tenncor::scatter(
					ETensor<T>(supcomp_grad), origshape, incrs);
			}
				break;
			case egen::SCATTER:
			{
				std::vector<teq::DimT> c;
				eigen::Packer<std::vector<teq::DimT>>().unpack(c, *op);

				std::vector<teq::DimT> strides;
				strides.reserve(teq::rank_cap);
				std::copy(c.begin(), c.begin() + std::min((size_t) teq::rank_cap, c.size()),
					std::back_inserter(strides));
				out = ETensor<T>(local_der) *
					tenncor::stride(ETensor<T>(supcomp_grad), strides);
			}
				break;
			case egen::REVERSE:
			{
				std::set<teq::RankT> dims;
				eigen::Packer<std::set<teq::RankT>>().unpack(dims, *op);

				out = ETensor<T>(local_der) * tenncor::reverse(ETensor<T>(supcomp_grad), dims);
			}
				break;
			case egen::SELECT:
			{
				if (0 == arg_idx)
				{
					out = ETensor<T>(local_der);
					break;
				}
				auto condition = ETensor<T>(
					op->get_children()[0]);
				auto then = ETensor<T>(supcomp_grad);
				auto otherwise = make_constant_scalar<T>(0, op->shape());
				if (1 < arg_idx)
				{
					std::swap(then, otherwise);
				}
				out = tenncor::if_then_else(condition, then, otherwise);
			}
				break;
			case egen::ARGMAX:
				teq::fatalf("cannot derive %s", opcode.name_.c_str());
				break;
			default:
				teq::fatalf("Unknown op %s", opcode.name_.c_str());
		}
		return (teq::TensptrT) out;
	}

	/// Implementation of iDerivativeFuncs
	teq::TensptrT get_const_one (teq::Shape shape) const override
	{
		return make_constant_scalar<T>(1, shape);
	}

	/// Implementation of iDerivativeFuncs
	teq::TensptrT get_const_zero (teq::Shape shape) const override
	{
		return make_constant_scalar<T>(0, shape);
	}

	/// Implementation of iDerivativeFuncs
	teq::TensptrT add (teq::TensptrsT elems) const override
	{
		assert(elems.size() > 0);
		return tenncor::sum(ETensorsT<T>(elems.begin(), elems.end()));
	}
};

/// Derive root with respect to target and optimized
template <typename T>
ETensor<T> derive (ETensor<T> root, ETensor<T> target)
{
	DerivativeFuncs<T> builder;
	teq::TensptrT derivative = teq::derive(root, target, builder);
	return ETensor<T>(derivative);
}

}

#endif // ETEQ_DERIVE_HPP
