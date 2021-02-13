///
/// backprop.hpp
/// eteq
///
/// Purpose:
/// Implement eteq definition for supporting backward propagation
///

#ifndef ETEQ_BACKPROP_HPP
#define ETEQ_BACKPROP_HPP

#include "internal/utils/coord/coord.hpp"

#include "tenncor/eteq/make.hpp"

namespace eteq
{

using EigenOpF = std::function<eigen::TensorT<size_t>(const eigen::TensMapT<size_t>&)>;

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

static inline teq::TensptrT elem_jacobianize (teq::TensptrT x, teq::TensptrT prev_chain)
{
	teq::Shape jacshape = prev_chain->shape();
	teq::DimT m = jacshape.at(0);
	teq::DimT n = jacshape.at(1);
	return make_functor(egen::EXTEND, {
		make_functor(egen::RESHAPE, {x}, teq::Shape({1, n}))
	}, teq::DimsT{m});
}

static teq::TensptrT lazy_jacobian (egen::_GENERATED_DTYPE dtype,
	const teq::Shape& outshape, const teq::Shape& inshape, EigenOpF manidx)
{
	teq::DimT m = inshape.n_elems();
	teq::DimT n = outshape.n_elems();
	teq::Shape jacshape({m, n});
	std::vector<size_t> is(m);
	std::iota(is.begin(), is.end(), 0);
	auto imap = eigen::make_tensmap(is.data(), inshape);
	auto indices = manidx(imap);
	size_t* idx = indices.data();
	std::vector<float> mat(m * n, 0);
	for (size_t y = 0; y < n; ++y)
	{
		if (idx[y] < m)
		{
			mat[y * m + idx[y]] = 1.f;
		}
	}
	return make_constant_tensor(mat.data(), jacshape, dtype);
}

static teq::TensptrT permute_jacobian (teq::RanksT order,
	teq::Shape outshape, teq::Shape inshape, egen::_GENERATED_DTYPE dtype)
{
	bool visited[teq::rank_cap];
	std::fill(visited, visited + teq::rank_cap, false);
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
	auto reorder = eigen::internal::dim_copy<
		teq::rank_cap,teq::RankT>(order);
	return lazy_jacobian(dtype, outshape, inshape,
	[reorder](const eigen::TensMapT<size_t>& index) ->
		eigen::TensorT<size_t>
	{
		return index.shuffle(reorder);
	});
}

static teq::TensptrT extend_jacobian (teq::DimsT bcast,
	teq::Shape outshape, teq::Shape inshape, egen::_GENERATED_DTYPE dtype)
{
	teq::ShapeT coord;
	std::fill(coord.begin(), coord.end(), 1);
	std::copy(bcast.begin(), bcast.begin() +
		std::min((size_t) teq::rank_cap, bcast.size()), coord.begin());
	return lazy_jacobian(dtype, outshape, inshape,
	[coord](const eigen::TensMapT<size_t>& index) ->
		eigen::TensorT<size_t>
	{
		return index.broadcast(coord);
	});
}

static teq::TensptrT rsum_jacobian (std::set<teq::RankT> ranks,
	teq::Shape outshape, teq::Shape inshape, egen::_GENERATED_DTYPE dtype)
{
	teq::RanksT vranks(ranks.begin(), ranks.end());
	teq::DimT m = inshape.n_elems();
	teq::DimT n = outshape.n_elems();
	teq::Shape jacshape({m, n});
	std::vector<float> mat(m * n, 0);
	for (size_t i = 0; i < m; ++i)
	{
		auto coord = teq::coordinate(inshape, i);
		for (auto rank : vranks)
		{
			coord[rank] = 0;
		}
		size_t j = teq::index(outshape, coord);
		mat[i + j * m] = 1;
	}
	return make_constant_tensor(mat.data(), jacshape, dtype);
}

static teq::TensptrT contract_jacobian (
	teq::Shape outshape, egen::_GENERATED_DTYPE dtype,
	teq::Shape lshape, teq::DimsT lexlist, teq::RanksT lpermlist,
	teq::TensptrT right, teq::DimsT rexlist, teq::RanksT rpermlist)
{
	marsh::Maps extmap;
	marsh::Maps permmap;
	eigen::pack_attr(extmap, lexlist);
	eigen::pack_attr(permmap, lpermlist);
	auto exshape = egen::ShapeParser<
		egen::EXTEND>()(extmap, {lshape});
	auto permshape = egen::ShapeParser<
		egen::PERMUTE>()(permmap, {exshape});

	auto ltrans = make_functor(egen::MATMUL, {
		permute_jacobian(
			lpermlist, permshape, exshape, dtype),
		extend_jacobian(
			lexlist, exshape, lshape, dtype)
	});
	auto rtrans = make_functor(egen::PERMUTE, {
		make_functor(egen::EXTEND, {right}, rexlist)
	}, rpermlist);
	auto out = make_functor(egen::MUL, {
		elem_jacobianize(rtrans, ltrans), ltrans
	});
	auto minrtrans = teq::narrow_shape(rtrans->shape());
	if (minrtrans.empty())
	{
		return out;
	}
	return make_functor(egen::MATMUL, {
		rsum_jacobian({minrtrans.size() - 1},
			outshape, rtrans->shape(), dtype), out
	});
}

static teq::TensptrT matmul_jacobian (teq::FuncptrT op, size_t arg_idx)
{
	auto args = op->get_args();
	auto ashape = args[0]->shape();
	auto bshape = args[1]->shape();
	auto cshape = op->shape();
	auto dtype = (egen::_GENERATED_DTYPE) op->get_meta().type_code();
	auto adim = ashape.at(1);
	auto bdim = bshape.at(0);
	teq::RankT an = teq::narrow_shape(ashape).size();
	teq::RankT bn = teq::narrow_shape(bshape).size();
	teq::DimsT aexlist(an, 1); aexlist.push_back(bdim);
	teq::RanksT apermlist = arrs::concat<teq::RankT>(
		{an, 1, 0}, arrs::range<teq::RankT>(2, an));
	teq::DimsT bexlist(bn, 1); bexlist.push_back(adim);
	teq::RanksT bpermlist = arrs::concat<teq::RankT>(
		{0, bn, 1}, arrs::range<teq::RankT>(2, bn));
	teq::TensptrT jacobian;
	if (0 == arg_idx)
	{
		jacobian = contract_jacobian(cshape, dtype,
			ashape, aexlist, apermlist,
			args[1], bexlist, bpermlist);
	}
	else // if (1 == arg_idx)
	{
		jacobian = contract_jacobian(cshape, dtype,
			bshape, bexlist, bpermlist,
			args[0], aexlist, apermlist);
	}
	return jacobian;
}

/// ETEQ implementation of TEQ's Backward Propagation Builder
struct DerivativeFuncs final : public teq::iBackpropFuncs
{
	teq::TensptrT jacobian_chain (teq::FuncptrT op,
		teq::TensptrT prev_chain, size_t arg_idx) const override
	{
		auto args = op->get_args();
		teq::Opcode opcode = op->get_opcode();
		teq::TensptrT out = nullptr;
		switch (opcode.code_)
		{
			case egen::IDENTITY:
			case egen::CAST:
			case egen::ROUND:
			case egen::ADD:
			case egen::RESHAPE:
				out = prev_chain;
				break;
			case egen::NEG:
				out = make_functor(egen::NEG, {prev_chain});
				break;
			case egen::TAN:
				out =  make_functor(egen::DIV, {
					prev_chain, elem_jacobianize(
						make_functor(egen::SQUARE, {
							make_functor(egen::COS, {args.front()}),
						}),
						prev_chain
					)
				});
				break;
			case egen::LOG:
				out = make_functor(egen::DIV, {
					prev_chain, elem_jacobianize(args.front(), prev_chain)
				});
				break;
			case egen::SQRT:
				out = make_functor(egen::DIV, {
					prev_chain, elem_jacobianize(
						make_functor(egen::MUL, {constant_like(2.f, op), op}),
						prev_chain
					)
				});
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
				teq::TensptrT dfx;
				switch (opcode.code_)
				{
					case egen::ABS:
						dfx = make_functor(egen::DIV, {args.front(), op});
						break;
					case egen::SIN:
						dfx = make_functor(egen::COS, {args.front()});
						break;
					case egen::COS:
						dfx = make_functor(egen::NEG, {
							make_functor(egen::SIN, {args.front()})});
						break;
					case egen::EXP:
						dfx = op;
						break;
					case egen::SQUARE:
						dfx = make_functor(egen::MUL, {
							constant_like(2.f, args.front()),
							args.front()
						});
						break;
					case egen::CUBE:
						dfx = make_functor(egen::MUL, {
							constant_like(3.f, args.front()),
							make_functor(egen::SQUARE, {args.front()}),
						});
						break;
					case egen::SIGMOID:
						dfx = make_functor(egen::MUL, {
							op, make_functor(egen::SUB, {
								constant_like(1.f, op), op
							})
						});
						break;
					case egen::TANH:
						dfx = make_functor(egen::SUB, {
							constant_like(1.f, op),
							make_functor(egen::SQUARE, {op}),
						});
						break;
					case egen::POW:
						dfx = arg_idx == 0 ? make_functor(egen::MUL, {
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
						dfx = make_functor(egen::MUL, nodes);
					}
						break;
					case egen::MAX:
					case egen::MIN:
						dfx = make_functor(egen::EQ, {op, args.at(arg_idx)});
						break;
				}
				out = make_functor(egen::MUL, {
					elem_jacobianize(dfx, prev_chain), prev_chain});
			}
				break;
			case egen::SUB:
				out = arg_idx == 0 ? prev_chain : make_functor(egen::NEG, {prev_chain});
				break;
			case egen::DIV:
				out = arg_idx == 0 ?
					make_functor(egen::DIV, {
						prev_chain,
						elem_jacobianize(args[1], prev_chain)
					}) :
					make_functor(egen::MUL, {
						make_functor(egen::NEG, {prev_chain}),
						elem_jacobianize(make_functor(egen::DIV,
							{op, args[1]}), prev_chain)
					});
				break;
			case egen::SELECT:
				if (0 == arg_idx)
				{
					out = get_const_zero(*prev_chain);
				}
				else if (1 == arg_idx)
				{
					out = make_functor(egen::SELECT, {
						elem_jacobianize(args[0], prev_chain),
						prev_chain, get_const_zero(*prev_chain)
					});
				}
				else // if (2 <= arg_idx)
				{
					out = make_functor(egen::SELECT, {
						elem_jacobianize(args[0], prev_chain),
						get_const_zero(*prev_chain), prev_chain
					});
				}
				break;
			case egen::RAND_UNIF:
			case egen::EQ:
			case egen::NEQ:
			case egen::GT:
			case egen::LT:
				out = get_const_zero(*prev_chain);
				break;
			// todo
			case egen::REVERSE:
			{
				// too lazy to figure out the actual transformation of reverse
				std::set<teq::RankT> dims;
				eigen::Packer<std::set<teq::RankT>>().unpack(dims, *op);
				std::array<bool,teq::rank_cap> do_reverse;
				std::fill(do_reverse.begin(), do_reverse.end(), false);
				for (teq::RankT i : dims)
				{
					do_reverse[i] = true;
				}
				out = make_functor(egen::MATMUL, {
					lazy_jacobian(
					(egen::_GENERATED_DTYPE) op->get_meta().type_code(),
					op->shape(), args.front()->shape(),
					[do_reverse](const eigen::TensMapT<size_t>& index) ->
						eigen::TensorT<size_t>
					{
						return index.reverse(do_reverse);
					}), prev_chain
				});
			}
				break;
			case egen::PERMUTE:
			{
				teq::RanksT order;
				eigen::Packer<teq::RanksT>().unpack(order, *op);
				auto jacobian = permute_jacobian(order,
						op->shape(), args.front()->shape(),
						(egen::_GENERATED_DTYPE) op->get_meta().type_code());
				out = make_functor(egen::MATMUL, {jacobian,
					prev_chain
				});
			}
				break;
			case egen::EXTEND:
			{
				teq::DimsT bcast = *eigen::unpack_extend(
					args.front()->shape(), *op);
				out = make_functor(egen::MATMUL, {
					extend_jacobian(bcast,
						op->shape(), args.front()->shape(),
						(egen::_GENERATED_DTYPE) op->get_meta().type_code()),
					prev_chain
				});
			}
				break;
			case egen::CONCAT:
			{
				auto outshape = op->shape();
				auto inshape = args[arg_idx]->shape();
				teq::RankT axis;
				eigen::Packer<teq::RankT>().unpack(axis, *op);
				teq::TensptrT jacobian;
				if (args.size() == 2)
				{
					if (arg_idx == 0)
					{
						auto bshape = args[1]->shape();
						jacobian = lazy_jacobian(
						(egen::_GENERATED_DTYPE) op->get_meta().type_code(),
						outshape, inshape,
						[outshape, bshape, arg_idx, axis](
							const eigen::TensMapT<size_t>& index) -> eigen::TensorT<size_t>
						{
							std::vector<size_t> ovec(
								bshape.n_elems(), outshape.n_elems());
							auto other = eigen::make_tensmap(ovec.data(), bshape);
							return index.concatenate(other,axis);
						});
					}
					else
					{
						auto ashape = args[0]->shape();
						jacobian = lazy_jacobian(
						(egen::_GENERATED_DTYPE) op->get_meta().type_code(),
						outshape, inshape,
						[outshape, ashape, arg_idx, axis](
							const eigen::TensMapT<size_t>& index) -> eigen::TensorT<size_t>
						{
							std::vector<size_t> ovec(
								ashape.n_elems(), outshape.n_elems());
							auto other = eigen::make_tensmap(ovec.data(), ashape);
							return other.concatenate(index,axis);
						});
					}
				}
				else
				{
					jacobian = lazy_jacobian(
					(egen::_GENERATED_DTYPE) op->get_meta().type_code(),
					outshape, inshape,
					[outshape, inshape, arg_idx, axis](
						const eigen::TensMapT<size_t>& index) -> eigen::TensorT<size_t>
					{
						std::vector<size_t> empties(
							outshape.n_elems(), inshape.n_elems());
						eigen::TensorT<size_t> out =
							eigen::make_tensmap<size_t>(empties.data(), outshape);
						std::array<Eigen::Index,teq::rank_cap-1> reshaped;
						auto outlist = outshape.to_list();
						auto it = outlist.begin();
						std::copy(it, it + axis, reshaped.begin());
						std::copy(it + axis + 1, outlist.end(),
							reshaped.begin() + axis);
						out.chip(arg_idx, axis) = index.reshape(reshaped);
						return out;
					});
				}
				out = make_functor(egen::MATMUL, {
					jacobian, prev_chain
				});
			}
				break;
			case egen::SLICE:
			{
				eigen::PairVecT<teq::DimT> encoding;
				eigen::Packer<eigen::PairVecT<teq::DimT>>().unpack(encoding, *op);
				teq::Shape shape = args.front()->shape();
				teq::ShapeT offsets;
				teq::ShapeT extents;
				std::fill(offsets.begin(), offsets.end(), 0);
				std::fill(extents.begin(), extents.end(), 1);
				std::copy(shape.begin(), shape.end(), extents.begin());
				size_t n = std::min(encoding.size(), (size_t) teq::rank_cap);
				for (size_t i = 0; i < n; ++i)
				{
					teq::DimT offset = std::min(encoding[i].first,
						(teq::DimT) (shape.at(i) - 1));
					offsets[i] = offset;
					extents[i] = std::min(encoding[i].second,
						(teq::DimT) (shape.at(i) - offset));
				}
				out = make_functor(egen::MATMUL, {
					lazy_jacobian(
					(egen::_GENERATED_DTYPE) op->get_meta().type_code(),
					op->shape(), shape,
					[offsets,extents](
						const eigen::TensMapT<size_t>& index) -> eigen::TensorT<size_t>
					{
						return index.slice(offsets, extents);
					}), prev_chain
				});
			}
				break;
			case egen::PAD:
			{
				eigen::PairVecT<teq::DimT> encoding;
				eigen::Packer<eigen::PairVecT<teq::DimT>>().unpack(encoding, *op);
				std::array<std::pair<teq::DimT,teq::DimT>,teq::rank_cap> paddings;
				std::fill(paddings.begin(), paddings.end(),
					std::pair<teq::DimT,teq::DimT>{0, 0});
				std::copy(encoding.begin(), encoding.begin() +
					std::min((size_t) teq::rank_cap, encoding.size()),
					paddings.begin());
				out = make_functor(egen::MATMUL, {
					lazy_jacobian(
					(egen::_GENERATED_DTYPE) op->get_meta().type_code(),
					op->shape(), args.front()->shape(),
					[paddings](
						const eigen::TensMapT<size_t>& index) -> eigen::TensorT<size_t>
					{
						return index.pad(paddings);
					}), prev_chain
				});
			}
				break;
			case egen::STRIDE:
			{
				teq::DimsT c;
				eigen::Packer<teq::DimsT>().unpack(c, *op);
				Eigen::array<Eigen::DenseIndex,teq::rank_cap> incrs;
				std::fill(incrs.begin(), incrs.end(), 1);
				std::copy(c.begin(), c.begin() + std::min((size_t) teq::rank_cap,
					c.size()), incrs.begin());
				out = make_functor(egen::MATMUL, {
					lazy_jacobian(
					(egen::_GENERATED_DTYPE) op->get_meta().type_code(),
					op->shape(), args.front()->shape(),
					[incrs](
						const eigen::TensMapT<size_t>& index) -> eigen::TensorT<size_t>
					{
						return index.stride(incrs);
					}), prev_chain
				});
			}
				break;
			case egen::SCATTER:
			{
				auto outshape = op->shape();
				auto inshape = args.front()->shape();
				teq::DimsT dims;
				eigen::Packer<teq::DimsT>().unpack(dims, *op);
				Eigen::array<Eigen::DenseIndex,teq::rank_cap> incrs;
				std::fill(incrs.begin(), incrs.end(), 1);
				std::copy(dims.begin(), dims.begin() +
					std::min((size_t) teq::rank_cap, dims.size()), incrs.begin());
				out = make_functor(egen::MATMUL, {
					lazy_jacobian(
					(egen::_GENERATED_DTYPE) op->get_meta().type_code(),
					outshape, inshape,
					[outshape, inshape, incrs](
						const eigen::TensMapT<size_t>& index) -> eigen::TensorT<size_t>
					{
						std::vector<size_t> empties(outshape.n_elems(),
							inshape.n_elems());
						eigen::TensorT<size_t> out =
							eigen::make_tensmap<size_t>(empties.data(), outshape);
						out.stride(incrs) = index;
						return out;
					}), prev_chain
				});
			}
				break;
			case egen::REDUCE_SUM:
			{
				std::set<teq::RankT> ranks;
				eigen::Packer<std::set<teq::RankT>>().unpack(ranks, *op);
				out = make_functor(egen::MATMUL, {
					rsum_jacobian(ranks,
						op->shape(), args.front()->shape(),
						(egen::_GENERATED_DTYPE) op->get_meta().type_code()),
					prev_chain
				});
			}
				break;
			case egen::REDUCE_PROD:
			{
				std::set<teq::RankT> ranks;
				eigen::Packer<std::set<teq::RankT>>().unpack(ranks, *op);
				auto arg = args.front();
				teq::RanksT vranks(ranks.begin(), ranks.end());
				auto outshape = op->shape();
				auto inshape = arg->shape();
				auto dtype = (egen::_GENERATED_DTYPE) op->get_meta().type_code();
				teq::DimT m = inshape.n_elems();
				teq::DimT n = outshape.n_elems();
				teq::Shape jxyshape({m, n});
				teq::Shape jxzshape({m, 1, m});
				std::vector<float> matxy(m * n, 0);
				std::vector<float> matxz(m * m, 1);
				std::vector<size_t> outbins[n];
				for (size_t i = 0; i < m; ++i)
				{
					auto coord = teq::coordinate(inshape, i);
					for (auto rank : vranks)
					{
						coord[rank] = 0;
					}
					size_t j = teq::index(outshape, coord);
					matxy[i + j * m] = 1;
					outbins[j].push_back(i);
				}
				for (size_t i = 0; i < n; ++i)
				{
					auto& bin = outbins[i];
					for (auto x : bin)
					{
						for (auto z : bin)
						{
							if (x != z)
							{
								matxz[x + z * m] = 0;
							}
						}
					}
				}
				auto xymask = make_constant_tensor(matxy.data(), jxyshape, dtype);
				auto xzmask = make_constant_tensor(matxz.data(), jxzshape, dtype);
				auto flatz = make_functor(egen::RESHAPE, {arg}, teq::Shape({1, 1, m}));
				auto xzplane = make_functor(egen::EXTEND, {flatz}, teq::DimsT{m});
				auto maskedxz = make_functor(egen::SELECT, {xzmask, xzmask, xzplane});
				auto flatx = make_functor(egen::REDUCE_PROD,
					{maskedxz}, std::set<teq::RankT>{2});
				auto xyplane = make_functor(egen::EXTEND, {flatx}, teq::DimsT{1, n});
				auto jacobian = make_functor(egen::MUL, {xymask, xyplane});
				out = make_functor(egen::MATMUL, {jacobian, prev_chain});
			}
				break;
			case egen::REDUCE_MAX:
			case egen::REDUCE_MIN:
			{
				std::set<teq::RankT> ranks;
				eigen::Packer<std::set<teq::RankT>>().unpack(ranks, *op);
				teq::RanksT vranks(ranks.begin(), ranks.end());
				auto arg = args.front();
				auto dtype = (egen::_GENERATED_DTYPE) op->get_meta().type_code();
				auto outshape = op->shape();
				auto inshape = arg->shape();
				teq::DimT m = inshape.n_elems();
				teq::DimT n = outshape.n_elems();
				teq::Shape jacshape({(teq::DimT) m, (teq::DimT) n});
				std::vector<float> mat(m * n, 0);
				for (size_t i = 0; i < m; ++i)
				{
					auto coord = teq::coordinate(inshape, i);
					for (auto rank : vranks)
					{
						coord[rank] = 0;
					}
					size_t j = teq::index(outshape, coord);
					mat[i + j * m] = 1.f;
				}
				auto flatx = make_functor(egen::RESHAPE, {arg}, teq::Shape({m}));
				auto flaty = make_functor(egen::RESHAPE, {op}, teq::Shape({1, n}));
				auto onemask = make_constant_tensor(mat.data(), jacshape, dtype);
				auto jacobian = make_functor(egen::MUL, {
					make_functor(egen::EQ, {
						make_functor(egen::EXTEND, {flatx}, teq::DimsT{1, n}),
						make_functor(egen::EXTEND, {flaty}, teq::DimsT{m})
					}),
					onemask
				});
				out = make_functor(egen::MATMUL, {jacobian, prev_chain});
			}
				break;
			case egen::MATMUL:
			{
				auto jacobian = matmul_jacobian(op, arg_idx);
				out = make_functor(egen::MATMUL, {jacobian, prev_chain});
			}
				break;
			case egen::CONTRACT:
			{
				eigen::PairVecT<teq::RankT> dims;
				eigen::Packer<eigen::PairVecT<teq::RankT>>().unpack(dims, *op);

				auto ashape = args[0]->shape();
				auto bshape = args[1]->shape();
				auto cshape = op->shape();
				auto dtype = (egen::_GENERATED_DTYPE) op->get_meta().type_code();
				teq::RankT an = teq::narrow_shape(ashape).size();
				teq::RankT bn = teq::narrow_shape(bshape).size();

				std::array<bool,teq::rank_cap> lvisit;
				std::array<bool,teq::rank_cap> rvisit;
				std::fill(lvisit.begin(), lvisit.end(), false);
				std::fill(rvisit.begin(), rvisit.end(), false);
				teq::RanksT lucom_ranks, rucom_ranks, lcom_ranks, rcom_ranks;
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
				teq::DimsT adims;
				for (teq::RankT i = 0; i < an; ++i)
				{
					if (false == lvisit[i])
					{
						lucom_ranks.push_back(i);
						adims.push_back(ashape.at(i));
					}
				}
				teq::DimsT bdims;
				for (teq::RankT i = 0 ; i < bn; ++i)
				{
					if (false == rvisit[i])
					{
						rucom_ranks.push_back(i);
						bdims.push_back(bshape.at(i));
					}
				}

				auto aexlist = arrs::concat(teq::DimsT(an, 1), bdims);
				teq::RanksT apermlist = arrs::concat(
					arrs::range<teq::RankT>(an, an + bdims.size()),
					lucom_ranks,
					lcom_ranks);
				auto bexlist = arrs::concat(teq::DimsT(bn, 1), adims);
				teq::RanksT bpermlist = arrs::concat<teq::RankT>(
					rucom_ranks,
					arrs::range<teq::RankT>(bn, bn + adims.size()),
					rcom_ranks);

				marsh::Maps extmap;
				marsh::Maps permmap;
				teq::TensptrT jacobian;
				if (0 == arg_idx)
				{
					jacobian = contract_jacobian(cshape, dtype,
						ashape, aexlist, apermlist,
						args[1], bexlist, bpermlist);
				}
				else // if (1 == arg_idx)
				{
					jacobian = contract_jacobian(cshape, dtype,
						bshape, bexlist, bpermlist,
						args[0], aexlist, apermlist);
				}
				out = make_functor(egen::MATMUL, {jacobian, prev_chain});
			}
				break;
			case egen::CONV:
			{
				// treat convolution as a matrix multiplication
				// then apply jacobian transformation
				teq::RanksT order;
				eigen::Packer<teq::RanksT>().unpack(order, *op);
				bool visited[teq::rank_cap];
				std::fill(visited, visited + teq::rank_cap, false);
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

				egen::ShapeParser<egen::PERMUTE> orderer;
				auto img = args[0];
				auto krn = args[1];
				auto ishape = orderer(*op, {img->shape()});
				auto kshape = krn->shape();
				auto cshape = orderer(*op, {op->shape()});
				auto dtype = (egen::_GENERATED_DTYPE) op->get_meta().type_code();

				teq::RankT minkrank = teq::narrow_shape(kshape).size();
				teq::NElemT nih1 = ishape.n_elems_between(0, minkrank);
				teq::NElemT nih2 = ishape.n_elems_between(minkrank, ishape.n_ranks());
				teq::NElemT nk = kshape.n_elems();
				teq::NElemT noh1 = cshape.n_elems_between(0, minkrank);
				// indices map
				teq::NElemT nkindices = noh1 * ishape.n_elems();
				teq::DimT shavex = 0;
				teq::NElemT nextra = 1;
				for (size_t i = 0; i < minkrank; ++i)
				{
					shavex += (ishape.at(i) - kshape.at(i)) * nextra;
					nextra *= ishape.at(i);
				}
				teq::NElemT nflat = nih1 - shavex;

				std::vector<size_t> projkern(nih1, 0);
				for (size_t i = 0; i < nk; ++i)
				{
					auto coord = teq::coordinate(kshape, i);
					size_t j = teq::index(ishape, coord);
					projkern[j] = i + 1;
				}
				auto projit = projkern.begin();
				std::vector<size_t> mink(projit, projit + nflat);
				std::vector<size_t> maskindices;
				for (size_t i = 0; i < noh1; ++i)
				{
					auto coord = teq::coordinate(cshape, i);
					auto prepad = teq::index(ishape, coord);
					auto postpad = nih1 - nflat - prepad;
					auto row = arrs::concat(
						std::vector<size_t>(prepad, 0),
						mink,
						std::vector<size_t>(postpad, 0));
					maskindices.insert(maskindices.end(), row.begin(), row.end());
				}
				std::vector<size_t> kindices;
				for (size_t i = 0; i < nih2; ++i)
				{
					kindices.insert(kindices.end(),
						maskindices.begin(), maskindices.end());
				}

				std::vector<float> kernelmask(nkindices * nk, 0);
				for (size_t i = 0; i < nkindices; ++i)
				{
					if (size_t j = kindices[i])
					{
						kernelmask[i * nk + j - 1] = 1;
					}
				}

				teq::DimsT halfi(ishape.begin() + minkrank, ishape.end());
				teq::Shape maskshape(teq::DimsT{nk, nkindices});
				auto mask = make_constant_tensor(
					kernelmask.data(), maskshape, dtype);
				auto transkin = make_functor(egen::MATMUL, {
					mask,
					make_functor(egen::RESHAPE, {krn}, teq::Shape({1, nk}))
				});
				teq::Shape transkshape(arrs::concat(teq::DimsT{nih1, noh1}, halfi));
				auto transk = make_functor(egen::RESHAPE, {transkin}, transkshape);

				teq::Shape transishape(arrs::concat(teq::DimsT{1, nih1}, halfi));
				auto transi = make_functor(egen::RESHAPE, {
					make_functor(egen::PERMUTE, {img}, order)
				}, transishape);

				teq::FuncptrT simulated_op = std::static_pointer_cast<
					teq::iFunctor>(make_functor(egen::MATMUL, {transk, transi}));

				auto jacobian = matmul_jacobian(simulated_op, arg_idx == 0);
				if (arg_idx == 0)
				{
					auto prm_jacobian = permute_jacobian(order,
						ishape, img->shape(), dtype);
					jacobian = make_functor(egen::MATMUL, {jacobian, prm_jacobian});
				}
				else
				{
					jacobian = make_functor(egen::MATMUL, {jacobian, mask});
				}
				out = make_functor(egen::MATMUL, {jacobian, prev_chain});
			}
				break;
			default:
				global::fatalf("Unsupported op derivation %s", opcode.name_.c_str());
		}
		return (teq::TensptrT) out;
	}

	teq::TensptrT dejacobianize (
		teq::TensptrT jacobian, teq::TensptrT x) const
	{
		teq::Shape outshape = x->shape();
		auto flatten = make_functor(egen::REDUCE_SUM, {jacobian},
			std::set<teq::RankT>{1});
		return make_functor(egen::RESHAPE, {flatten}, outshape);
	}

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

	teq::TensptrT get_const_eye (teq::NElemT n, size_t type_code) const override
	{
		auto reftype = (egen::_GENERATED_DTYPE) type_code;
		teq::Shape shape({(teq::DimT) n, (teq::DimT) n});
		std::vector<float> data(shape.n_elems(), 0.f);
		for (size_t i = 0; i < n; ++i)
		{
			data[i + n * i] = 1.f;
		}
		return make_constant_tensor(data.data(), shape, reftype);
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

	teq::TensptrT constant (float scalar, teq::Shape shape,
		egen::_GENERATED_DTYPE dtype) const
	{
		std::vector<float> svec(shape.n_elems(), scalar);
		return make_constant_tensor(svec.data(), shape, dtype);
	}
};

}

#endif // ETEQ_BACKPROP_HPP
