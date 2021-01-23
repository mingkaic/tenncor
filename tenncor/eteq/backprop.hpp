///
/// backprop.hpp
/// eteq
///
/// Purpose:
/// Implement eteq definition for supporting backward propagation
///

#ifndef ETEQ_BACKPROP_HPP
#define ETEQ_BACKPROP_HPP

#include "tenncor/eteq/make.hpp"

namespace eteq
{

/// Return reduction operator gradient of reduced functor node (bwd)
static inline teq::TensptrT reduce_grad (teq::Shape shape, teq::TensptrT bwd, teq::FuncptrT fwd)
{
	teq::DimsT bcast(teq::rank_cap, 1);
	std::set<teq::RankT> ranks;
	eigen::Packer<std::set<teq::RankT>>().unpack(ranks, *fwd);
	for (teq::RankT d : ranks)
	{
		if (d < teq::rank_cap)
		{
			bcast[d] = shape.at(d);
		}
	}
	return make_functor(egen::EXTEND, {bwd}, bcast);
}

static inline teq::RanksT reorder_permute (
	teq::RanksT order)
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
	teq::RanksT reorder(teq::rank_cap);
	for (size_t i = 0; i < teq::rank_cap; ++i)
	{
		reorder[order[i]] = i;
	}
	return reorder;
}

/// ETEQ implementation of TEQ's Backward Propagation Builder
struct DerivativeFuncs final : public teq::iDerivativeFuncs
{
	/// Implementation of iDerivativeFuncs
	teq::TensptrT lderive (teq::FuncptrT op,
		teq::TensptrT supgrad, size_t arg_idx) const override
	{
		auto args = op->get_args();
		teq::Opcode opcode = op->get_opcode();
		teq::TensptrT out;
		switch (opcode.code_)
		{
			case egen::IDENTITY:
			case egen::CAST:
			case egen::ROUND:
			case egen::ADD:
				out = supgrad;
				break;
			case egen::NEG:
				out = make_functor(egen::NEG, {supgrad});
				break;
			case egen::TAN:
				out =  make_functor(egen::DIV, {
					supgrad,
					make_functor(egen::SQUARE, {
						make_functor(egen::COS, {args.front()}),
					})
				});
				break;
			case egen::LOG:
				out = make_functor(egen::DIV, {
					supgrad, args.front()});
				break;
			case egen::SQRT:
				out = make_functor(egen::DIV, {
					supgrad, make_functor(egen::MUL, {
						constant_like(2.f, op), op})});
				break;
			case egen::ABS:
			case egen::SIN:
			case egen::COS:
			case egen::EXP:
			case egen::SQUARE:
			case egen::CUBE:
			case egen::SIGMOID:
			case egen::TANH:
			case egen::POW:
			case egen::MUL:
			case egen::MAX:
			case egen::MIN:
			{
				teq::TensptrT local_der;
				switch (opcode.code_)
				{
					case egen::ABS:
						local_der = make_functor(egen::DIV, {args.front(), op});
						break;
					case egen::SIN:
						local_der = make_functor(egen::COS, {args.front()});
						break;
					case egen::COS:
						local_der = make_functor(egen::NEG, {
							make_functor(egen::SIN, {args.front()})});
						break;
					case egen::EXP:
						local_der = op;
						break;
					case egen::SQUARE:
						local_der = make_functor(egen::MUL, {
							constant_like(2.f, args.front()),
							args.front()
						});
						break;
					case egen::CUBE:
						local_der = make_functor(egen::MUL, {
							constant_like(3.f, args.front()),
							make_functor(egen::SQUARE, {args.front()}),
						});
						break;
					case egen::SIGMOID:
						local_der = make_functor(egen::MUL, {
							op, make_functor(egen::SUB, {
								constant_like(1.f, op), op
							})
						});
						break;
					case egen::TANH:
						local_der = make_functor(egen::SUB, {
							constant_like(1.f, op),
							make_functor(egen::SQUARE, {op}),
						});
						break;
					case egen::POW:
						local_der = arg_idx == 0 ? make_functor(egen::MUL, {
								args[1], make_functor(egen::POW, {
									args[0], make_functor(egen::SUB, {
										args[1], constant_like(1.f, args[1])
									})
								})
							}) :
							make_functor(egen::MUL, {
								make_functor(egen::LOG, {args.front()}), op});
						break;
					case egen::MUL:
					{
						size_t nargs = args.size();
						teq::TensptrsT nodes;
						nodes.reserve(nargs);
						for (size_t i = 0, n = nargs; i < n; ++i)
						{
							if (i != arg_idx)
							{
								nodes.push_back(args[i]);
							}
						}
						local_der = make_functor(egen::MUL, nodes);
					}
						break;
					case egen::MAX:
					case egen::MIN:
						local_der = make_functor(egen::EQ, {op, args.at(arg_idx)});
						break;
				}
				out = make_functor(egen::MUL, {local_der, supgrad});
			}
				break;
			case egen::SUB:
				out = arg_idx == 0 ? supgrad : make_functor(egen::NEG, {supgrad});
				break;
			case egen::DIV:
				out = arg_idx == 0 ? make_functor(egen::DIV, {supgrad, args[1]}) :
					make_functor(egen::DIV, {
						make_functor(egen::DIV, {
							make_functor(egen::MUL, {
								make_functor(egen::NEG, {supgrad}), args[0]}),
							args[1]
						}), args[1]
					});
				break;
			case egen::REDUCE_SUM:
				out = reduce_grad(args.front()->shape(), supgrad, op);
				break;
			case egen::REDUCE_PROD:
				out = make_functor(egen::MUL, {
					reduce_grad(args.front()->shape(), supgrad, op),
					make_functor(egen::DIV, {
						reduce_grad(args.front()->shape(), op, op),
						args.front(),
					})
				});
				break;
			case egen::REDUCE_MAX:
			case egen::REDUCE_MIN:
				out = make_functor(egen::EQ, {
					reduce_grad(args.front()->shape(), op, op),
					make_functor(egen::MUL, {
						args.front(),
						reduce_grad(args.front()->shape(), supgrad, op),
					})
				});
				break;
			case egen::EXTEND:
			{
				teq::DimsT bcast = *eigen::unpack_extend(
					args.front()->shape(), *op);

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
				out = make_functor(egen::REDUCE_SUM, {supgrad}, dims);
			}
				break;
			case egen::PERMUTE:
			{
				teq::RanksT order;
				eigen::Packer<teq::RanksT>().unpack(order, *op);

				out = make_functor(egen::PERMUTE, {supgrad}, reorder_permute(order));
			}
				break;
			case egen::RESHAPE:
			{
				out = make_functor(egen::RESHAPE, {supgrad}, args.front()->shape());
			}
				break;
			case egen::MATMUL:
				if (arg_idx == 0)
				{
					out = make_functor(egen::MATMUL, {
						supgrad, 
						make_functor(egen::PERMUTE, {args[1]}, teq::RanksT{1, 0})
					});
				}
				else
				{
					// (sup^T @ arg0)^T = arg0^T @ sup
					out = make_functor(egen::MATMUL, {
						make_functor(egen::PERMUTE, {args[0]}, teq::RanksT{1, 0}),
						supgrad 
					});
				}
				break;
			case egen::CONTRACT:
			{
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
				teq::RanksT
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

				// left = supgrad
				teq::TensptrT right;
				teq::RanksT order;
				eigen::PairVecT<teq::RankT> grad_dims;
				if (arg_idx == 0)
				{
					right = args[1];
					for (teq::RankT i = 0, n = rucom_ranks.size(); i < n; ++i)
					{
						grad_dims.push_back({i, rucom_ranks[i]});
					}
					// contract output has shape <lucom, lcom>
					order = lcom_ranks;
					order.insert(order.end(), lucom_ranks.begin(), lucom_ranks.end());
					// reverse order such that contract output permutes to args[0]->shape()
					order = reorder_permute(order);
				}
				else
				{
					right = args[0];
					for (teq::RankT i = 0, n = lucom_ranks.size(); i < n; ++i)
					{
						grad_dims.push_back({rucom_ranks.size() + i, lucom_ranks[i]});
					}
					// contract output has shape <rcom, rucom>
					order = rcom_ranks;
					order.insert(order.end(), rucom_ranks.begin(), rucom_ranks.end());
					// reverse order such that contract output permutes to args[1]->shape()
					order = reorder_permute(order);
				}
				if (grad_dims.empty())
				{
					grad_dims.push_back({
						teq::narrow_shape(supgrad->shape()).size(),
						teq::narrow_shape(right->shape()).size()});
				}
				out = make_functor(egen::PERMUTE, {
					make_functor(egen::CONTRACT, {supgrad, right}, grad_dims),
				}, order);
			}
				break;
			case egen::CONV:
			{
				// for convolution(X, Y) = C
				teq::RanksT order;
				eigen::Packer<teq::RanksT>().unpack(order, *op);

				teq::RanksT dims;
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
					teq::RanksT revdims(ndims);
					std::iota(revdims.begin(), revdims.end(), 0);
					out = make_functor(egen::CONV, {
						make_functor(egen::PAD, {supgrad}, paddings),
						make_functor(egen::REVERSE, {args[1]},
							std::set<teq::RankT>(revdims.begin(), revdims.end()))
					}, dims);
				}
				else
				{
					// convolve(X, C_grad_sup)
					teq::RanksT indices(teq::rank_cap);
					std::iota(indices.begin(), indices.end(), 0);
					out = make_functor(egen::PERMUTE, {
						make_functor(egen::CONV, {args[0], supgrad}, indices)
					}, dims);
				}
			}
				break;
			case egen::SLICE:
			{
				eigen::PairVecT<teq::DimT> extents;
				eigen::Packer<eigen::PairVecT<teq::DimT>>().unpack(extents, *op);

				teq::Shape cshape = args.front()->shape();
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
				out = make_functor(egen::PAD, {supgrad}, paddings);
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
				out = make_functor(egen::SLICE, {supgrad}, extents);
			}
				break;
			case egen::CONCAT:
			{
				auto children = args;
				teq::Shape cshape = children[arg_idx]->shape();
				teq::RankT axis;
				eigen::Packer<teq::RankT>().unpack(axis, *op);
				if (children.size() > 2)
				{
					eigen::PairVecT<teq::DimT> extents(
						std::max(teq::rank_cap, axis),
						{0,std::numeric_limits<teq::DimT>::max()});
					extents[axis] = {arg_idx, 1};
					out = make_functor(egen::SLICE, {supgrad}, extents);
				}
				else
				{
					teq::DimT offset = 0;
					teq::DimT extent = cshape.at(axis);
					if (arg_idx)
					{
						teq::Shape first_shape = children[0]->shape();

						offset = first_shape.at(axis);
					}
					eigen::PairVecT<teq::DimT> extents(
						std::max(teq::rank_cap, axis),
						{0,std::numeric_limits<teq::DimT>::max()});
					extents[axis] = {offset, extent};
					out = make_functor(egen::SLICE, {supgrad}, extents);
				}
			}
				break;
			case egen::STRIDE:
			{
				teq::DimsT incrs;
				eigen::Packer<teq::DimsT>().unpack(incrs, *op);

				teq::Shape origshape = args[0]->shape();
				out = make_functor(egen::SCATTER, {supgrad}, origshape, incrs);
			}
				break;
			case egen::SCATTER:
			{
				teq::DimsT c;
				eigen::Packer<teq::DimsT>().unpack(c, *op);

				teq::DimsT strides;
				strides.reserve(teq::rank_cap);
				std::copy(c.begin(), c.begin() + std::min((size_t) teq::rank_cap, c.size()),
					std::back_inserter(strides));
				out = make_functor(egen::STRIDE, {supgrad}, strides);
			}
				break;
			case egen::REVERSE:
			{
				std::set<teq::RankT> dims;
				eigen::Packer<std::set<teq::RankT>>().unpack(dims, *op);

				out = make_functor(egen::REVERSE, {supgrad}, dims);
			}
				break;
			case egen::SELECT:
			{
				if (0 == arg_idx)
				{
					out = constant_like(0.f, args.front());
					break;
				}
				teq::TensptrT condition = args[0];
				teq::TensptrT then, otherwise;
				if (arg_idx == 1)
				{
					then = supgrad;
					otherwise = constant_like(0.f, op);
				}
				else // if (arg_idx > 2)
				{
					then = constant_like(0.f, op);
					otherwise = supgrad;
				}
				out = make_functor(egen::SELECT, {condition, then, otherwise});
			}
				break;
			case egen::RAND_UNIF:
			case egen::EQ:
			case egen::NEQ:
			case egen::GT:
			case egen::LT:
				out = constant_like(0.f, args.front());
				break;
			case egen::ASSIGN:
			case egen::ASSIGN_ADD:
			case egen::ASSIGN_SUB:
			case egen::ASSIGN_MUL:
			case egen::ASSIGN_DIV:
			case egen::ARGMAX:
				global::fatalf("cannot derive %s", opcode.name_.c_str());
				break;
			default:
				global::fatalf("Unknown op %s", opcode.name_.c_str());
		}
		return (teq::TensptrT) out;
	}

	/// Implementation of iDerivativeFuncs
	teq::TensptrT get_const_one (teq::iTensor& reference) const override
	{
		auto reftype = (egen::_GENERATED_DTYPE) reference.get_meta().type_code();
		auto shape = reference.shape();
		std::vector<float> data(shape.n_elems(), 1.f);
		return make_constant_tensor(data.data(), shape, reftype);
	}

	/// Implementation of iDerivativeFuncs
	teq::TensptrT get_const_zero (teq::iTensor& reference) const override
	{
		auto reftype = (egen::_GENERATED_DTYPE) reference.get_meta().type_code();
		auto shape = reference.shape();
		std::vector<float> data(shape.n_elems(), 0.f);
		return make_constant_tensor(data.data(), shape, reftype);
	}

	/// Implementation of iDerivativeFuncs
	teq::TensptrT add (teq::TensptrsT elems) const override
	{
		assert(elems.size() > 0);
		return make_functor(egen::ADD, elems);
	}

private:
	teq::TensptrT constant_like (float scalar, teq::TensptrT like) const
	{
		auto like_type = (egen::_GENERATED_DTYPE) like->get_meta().type_code();
		teq::TensptrT cst = make_constant_tensor(&scalar, teq::Shape(), like_type);
		return make_functor(::egen::EXTEND, teq::TensptrsT{cst}, (teq::TensptrT) like);
	}
};

}

#endif // ETEQ_BACKPROP_HPP
