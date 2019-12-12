///
/// grader.hpp
/// eteq
///
/// Purpose:
/// Implement eteq gradient definition for supported operations
///

#include "teq/grad_def.hpp"

#include "eigen/operator.hpp"

#include "eteq/generated/api.hpp"
#include "eteq/constant.hpp"

#ifndef ETEQ_GRADER_HPP
#define ETEQ_GRADER_HPP

namespace eteq
{

/// Return reduction operator gradient of reduced functor node (bwd)
template <typename T>
LinkptrT<T> reduce_grad (teq::Shape shape,
	LinkptrT<T> bwd, teq::FuncptrT fwd)
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

/// ETEQ implementation of TEQ's Backward Propagation Builder
template <typename T>
struct GradientBuilder final : public teq::iGradientBuilder
{
	/// Implementation of iGradientBuilder
	teq::TensptrT local_derivative (teq::FuncptrT op,
		size_t arg_idx) const override
	{
		auto args = op->get_children();
		LinkptrT<T> out = nullptr;
		teq::Opcode opcode = op->get_opcode();
		switch ((egen::_GENERATED_OPCODE) opcode.code_)
		{
			case egen::ABS:
				out = to_link<T>(args[0]) / to_link<T>(op);
				break;
			case egen::NEG:
				out = make_constant_scalar<T>(
					-1, args[0]->shape());
				break;
			case egen::SIN:
				out = tenncor::cos(to_link<T>(args[0]));
				break;
			case egen::COS:
				out = -tenncor::sin(to_link<T>(args[0]));
				break;
			case egen::TAN:
				out = (T) 1 / tenncor::pow(
					tenncor::cos(to_link<T>(args[0])), (T) 2);
				break;
			case egen::EXP:
				out = to_link<T>(op);
				break;
			case egen::LOG:
				out = (T) 1 / to_link<T>(args[0]);
				break;
			case egen::SQRT:
				out = (T) 1 / ((T) 2 * to_link<T>(op));
				break;
			case egen::SQUARE:
				out = (T) 2 * to_link<T>(args[0]);
				break;
			case egen::CUBE:
				out = (T) 3 * tenncor::square(to_link<T>(args[0]));
				break;
			case egen::SIGMOID:
				out = to_link<T>(op) * ((T) 1 - to_link<T>(op));
				break;
			case egen::TANH:
				out = (T) 1 - tenncor::square(to_link<T>(op));
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
				out = make_constant_scalar<T>(1, args[arg_idx]->shape());
				break;
			case egen::MUL:
			case egen::GROUP_PROD:
			{
				LinksT<T> nodes;
				size_t nargs = args.size();
				nodes.reserve(nargs);
				for (size_t i = 0, n = nargs; i < n; ++i)
				{
					if (i != arg_idx)
					{
						nodes.push_back(to_link<T>(args[i]));
					}
				}
				out = tenncor::prod(nodes);
			}
				break;
			case egen::MAX:
			case egen::MIN:
				out = to_link<T>(op) == to_link<T>(args[arg_idx]);
				break;
			case egen::POW:
				out = arg_idx==0 ?
					to_link<T>(args[1]) *
					tenncor::pow(
						to_link<T>(args[0]),
						to_link<T>(args[1]) - (T) 1
					) :
					tenncor::log(to_link<T>(args[0])) *
						to_link<T>(op);
				break;
			case egen::SUB:
				out = make_constant_scalar<T>(arg_idx == 0 ?
					1 : -1, args[0]->shape());
				break;
			case egen::DIV:
			{
				auto denom = to_link<T>(args[1]);
				out = arg_idx==0 ?
					(T) 1 / denom :
					-to_link<T>(args[0]) / denom / denom;
			}
				break;
			case egen::EQ:
			case egen::NEQ:
			case egen::GT:
			case egen::LT:
			case egen::RAND_UNIF:
			case egen::SELECT:
				out = make_constant_scalar<T>(0, args[0]->shape());
				break;
			case egen::REDUCE_PROD: // todo: prevent divide by zero
				out =
					reduce_grad(args[0]->shape(), to_link<T>(op), op) /
					to_link<T>(args[0]);
				break;
			case egen::REDUCE_MAX:
			case egen::REDUCE_MIN:
				out =
					reduce_grad(args[0]->shape(), to_link<T>(op), op) ==
					to_link<T>(args[0]);
				break;
			case egen::MATMUL:
			{
				LinkptrT<T> lhs = to_link<T>(args[0]);
				LinkptrT<T> rhs = to_link<T>(args[1]);

				eigen::PairVecT<teq::RankT> dims;
				eigen::Packer<eigen::PairVecT<teq::RankT>>().unpack(dims, *op);

				std::vector<teq::DimT> llist = teq::narrow_shape(lhs->shape());
				std::vector<teq::DimT> rlist = teq::narrow_shape(rhs->shape());
				std::array<bool,teq::rank_cap> avisit;
				std::array<bool,teq::rank_cap> bvisit;
				std::fill(avisit.begin(), avisit.end(), false);
				std::fill(bvisit.begin(), bvisit.end(), false);
				for (auto coms : dims)
				{
					teq::RankT ldim = coms.first;
					teq::RankT rdim = coms.second;
					avisit[ldim] = true;
					bvisit[rdim] = true;
					// append 1s for mapped common dimensions
					if (ldim > llist.size())
					{
						llist.insert(llist.end(), ldim - llist.size(), 1);
					}
					if (rdim > rlist.size())
					{
						rlist.insert(rlist.end(), rdim - rlist.size(), 1);
					}
				}
				std::vector<teq::RankT> lcommon, luncommon, rcommon, runcommon;
				for (teq::RankT i = 0, n = llist.size(); i < n; ++i)
				{
					(avisit[i] ? &lcommon : &luncommon)->push_back(i);
				}
				for (teq::RankT i = 0, n = rlist.size(); i < n; ++i)
				{
					(bvisit[i] ? &rcommon : &runcommon)->push_back(i);
				}

				// desired shape as follows:
				// uncommon dimensions go in front ordered by <right uncommon><left uncommon><common>
				LinkptrT<T> ext;
				std::vector<teq::RankT> permlist;
				if (0 == arg_idx)
				{
					// extend rhs to fit before reduce summation to op:
					// rshape contains: <right uncommon + common>, so extend with <left uncommon>
					// <right uncommon + common><left uncommon> permute to desired shape

					std::vector<teq::DimT> luncom_dims;
					luncom_dims.reserve(luncommon.size());
					std::transform(luncommon.begin(), luncommon.end(),
						std::back_inserter(luncom_dims),
						[&llist](teq::RankT i)
						{
							return llist.at(i);
						});
					teq::RankT roffset = rlist.size();
					ext = tenncor::extend(rhs, roffset, luncom_dims);
					permlist = runcommon;
					permlist.reserve(luncommon.size());
					for (teq::RankT i = 0, n = luncommon.size(); i < n; ++i)
					{
						permlist.push_back(i + roffset);
					}
					permlist.insert(permlist.end(), rcommon.begin(), rcommon.end());
				}
				else
				{
					// extend rhs to fit before reduce summation to op:
					// lshape contains: <left uncommon + common>, so extend with <right uncommon>
					// <left uncommon + common><right uncommon> permute to desired shape

					std::vector<teq::DimT> runcom_dims;
					runcom_dims.reserve(runcommon.size());
					std::transform(runcommon.begin(), runcommon.end(),
						std::back_inserter(runcom_dims),
						[&rlist](teq::RankT i)
						{
							return rlist.at(i);
						});
					teq::RankT loffset = llist.size();
					ext = tenncor::extend(lhs, loffset, runcom_dims);
					permlist.reserve(runcommon.size());
					for (teq::RankT i = 0, n = runcommon.size(); i < n; ++i)
					{
						permlist.push_back(i + loffset);
					}
					permlist.insert(permlist.end(), luncommon.begin(), luncommon.end());
					permlist.insert(permlist.end(), lcommon.begin(), lcommon.end());
				}
				out = tenncor::permute(ext, permlist);
			}
				break;
			case egen::ARGMAX:
				logs::fatalf("cannot derive %s", opcode.name_.c_str());
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
		LinkptrT<T> out = nullptr;
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
				out = to_link<T>(local_der) *
					to_link<T>(supcomp_grad);
				break;
			case egen::REDUCE_MAX:
			case egen::REDUCE_MIN:
			case egen::REDUCE_PROD:
			case egen::REDUCE_SUM:
				out = to_link<T>(local_der) * reduce_grad(
					op->get_children()[0]->shape(),
					to_link<T>(supcomp_grad), op);
				break;
			case egen::EXTEND:
			{
				std::vector<teq::DimT> bcast;
				eigen::Packer<std::vector<teq::DimT>>().unpack(bcast, *op);

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
				out = to_link<T>(local_der) * tenncor::reduce_sum(to_link<T>(supcomp_grad), dims);
			}
				break;
			case egen::PERMUTE:
			{
				std::vector<teq::RankT> order;
				eigen::Packer<std::vector<teq::RankT>>().unpack(order, *op);

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

				std::vector<teq::RankT> reorder(teq::rank_cap);
				for (size_t i = 0; i < teq::rank_cap; ++i)
				{
					reorder[order[i]] = i;
				}
				out = to_link<T>(local_der) * tenncor::permute(
					to_link<T>(supcomp_grad), reorder);
			}
				break;
			case egen::RESHAPE:
			{
				out = to_link<T>(local_der) * tenncor::reshape(
					to_link<T>(supcomp_grad),
					op->get_children()[0]->shape());
			}
				break;
			case egen::MATMUL:
			{
				auto args = op->get_children();
				eigen::PairVecT<teq::RankT> dims;
				eigen::Packer<eigen::PairVecT<teq::RankT>>().unpack(dims, *op);

				std::vector<teq::DimT> llist = teq::narrow_shape(args[0]->shape());
				std::vector<teq::DimT> rlist = teq::narrow_shape(args[1]->shape());
				std::array<bool,teq::rank_cap> avisit;
				std::array<bool,teq::rank_cap> bvisit;
				std::fill(avisit.begin(), avisit.end(), false);
				std::fill(bvisit.begin(), bvisit.end(), false);
				for (auto coms : dims)
				{
					teq::RankT ldim = coms.first;
					teq::RankT rdim = coms.second;
					avisit[ldim] = true;
					bvisit[rdim] = true;
					// append 1s for mapped common dimensions
					if (ldim > llist.size())
					{
						llist.insert(llist.end(), ldim - llist.size(), 1);
					}
					if (rdim > rlist.size())
					{
						rlist.insert(rlist.end(), rdim - rlist.size(), 1);
					}
				}
				std::vector<teq::RankT> lcommon, luncommon, rcommon, runcommon;
				for (teq::RankT i = 0, n = llist.size(); i < n; ++i)
				{
					(avisit[i] ? &lcommon : &luncommon)->push_back(i);
				}
				for (teq::RankT i = 0, n = rlist.size(); i < n; ++i)
				{
					(bvisit[i] ? &rcommon : &runcommon)->push_back(i);
				}

				std::vector<teq::DimT> extlist;
				if (0 == arg_idx)
				{
					extlist.reserve(lcommon.size());
					for (teq::RankT l : lcommon)
					{
						extlist.push_back(llist.at(l));
					}
				}
				else
				{
					extlist.reserve(rcommon.size());
					for (teq::RankT r : rcommon)
					{
						extlist.push_back(rlist.at(r));
					}
				}

				// desired shape as follows:
				// uncommon dimensions go in front ordered by <supcomp_grad shape><common of arg_idx>
				// <supcomp_grad shape> = <right uncommon><left uncommon>
				auto ext = tenncor::extend(to_link<T>(supcomp_grad),
					luncommon.size() + runcommon.size(), extlist);

				// todo: apply dims
				teq::RankT arg_rank;
				std::vector<teq::RankT> fwdperm;
				if (0 == arg_idx)
				{
					// currently: <right uncommon><left uncommon><left common>
					// permute such that shape = <left shape><uncommon right>
					arg_rank = llist.size();
					for (teq::RankT i = 0, n = runcommon.size(); i < n; ++i)
					{
						fwdperm.push_back(arg_rank + i);
					}
					fwdperm.insert(fwdperm.end(), luncommon.begin(), luncommon.end());
					fwdperm.insert(fwdperm.end(), lcommon.begin(), lcommon.end());
				}
				else
				{
					// currently: <right uncommon><left uncommon><right common>
					// permute such that shape = <right shape><uncommon left>
					arg_rank = rlist.size();
					fwdperm = runcommon;
					for (teq::RankT i = 0, n = luncommon.size(); i < n; ++i)
					{
						fwdperm.push_back(arg_rank + i);
					}
					fwdperm.insert(fwdperm.end(), rcommon.begin(), rcommon.end());
				}
				// reverse fwdperm
				size_t nperms = fwdperm.size();
				std::vector<teq::RankT> permlist(nperms);
				for (size_t i = 0; i < nperms; ++i)
				{
					permlist[fwdperm[i]] = i;
				}

				// reduce <uncommon not arg_idx> such that shape = <arg_idx shape>
				out = tenncor::reduce_sum(
					tenncor::permute(to_link<T>(local_der) * ext, permlist),
					arg_rank, teq::rank_cap - arg_rank);
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
						to_link<T>(supcomp_grad), paddings),
						tenncor::reverse(
							to_link<T>(args[1]),
							std::set<teq::RankT>(revdims.begin(), revdims.end())), dims);
				}
				else
				{
					// convolve(X, C_grad_sup)
					std::vector<teq::RankT> indices(teq::rank_cap);
					std::iota(indices.begin(), indices.end(), 0);
					out = tenncor::permute(
						tenncor::convolution(
							to_link<T>(args[0]),
							to_link<T>(supcomp_grad),
							indices), dims);
				}
			}
				break;
			case egen::SLICE:
			{
				eigen::PairVecT<teq::DimT> extents;
				eigen::Packer<eigen::PairVecT<teq::DimT>>().unpack(extents, *op);

				teq::Shape cshape = op->get_children()[0]->shape();
				eigen::PairVecT<teq::DimT> paddings;
				paddings.reserve(teq::rank_cap);
				for (size_t i = 0; i < teq::rank_cap; ++i)
				{
					teq::DimT leftpad = extents[i].first;
					paddings.push_back({leftpad,
						cshape.at(i) - (leftpad + extents[i].second)});
				}
				out = to_link<T>(local_der) *
					tenncor::pad(to_link<T>(supcomp_grad), paddings);
			}
				break;
			case egen::PAD:
			{
				eigen::PairVecT<teq::DimT> paddings;
				eigen::Packer<eigen::PairVecT<teq::DimT>>().unpack(paddings, *op);

				teq::Shape oshape = op->shape();
				eigen::PairVecT<teq::DimT> extents;
				extents.reserve(teq::rank_cap);
				for (size_t i = 0; i < teq::rank_cap; ++i)
				{
					teq::DimT offset = paddings[i].first;
					extents.push_back({offset,
						oshape.at(i) - paddings[i].second - offset});
				}
				out = to_link<T>(local_der) *
					tenncor::slice(to_link<T>(supcomp_grad), extents);
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
				out = to_link<T>(local_der) *
					tenncor::slice(to_link<T>(supcomp_grad), offset, extent, axis);
			}
				break;
			case egen::GROUP_CONCAT: // todo: combine concat and group_concat
			{
				teq::RankT axis;
				eigen::Packer<teq::RankT>().unpack(axis, *op);
				out = to_link<T>(local_der) *
					tenncor::slice(to_link<T>(supcomp_grad), arg_idx, 1, axis);
			}
				break;
			case egen::STRIDE:
			{
				std::vector<teq::DimT> incrs;
				eigen::Packer<std::vector<teq::DimT>>().unpack(incrs, *op);

				teq::Shape origshape = op->get_children()[0]->shape();
				out = to_link<T>(local_der) * tenncor::scatter(
					to_link<T>(supcomp_grad), origshape, incrs);
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
				out = to_link<T>(local_der) *
					tenncor::stride(to_link<T>(supcomp_grad), strides);
			}
				break;
			case egen::REVERSE:
			{
				std::set<teq::RankT> dims;
				eigen::Packer<std::set<teq::RankT>>().unpack(dims, *op);

				out = to_link<T>(local_der) * tenncor::reverse(to_link<T>(supcomp_grad), dims);
			}
				break;
			case egen::SELECT:
			{
				if (0 == arg_idx)
				{
					out = to_link<T>(local_der);
					break;
				}
				auto condition = to_link<T>(
					op->get_children()[0]);
				auto then = to_link<T>(supcomp_grad);
				auto otherwise = make_constant_scalar<T>(0, op->shape());
				if (1 < arg_idx)
				{
					std::swap(then, otherwise);
				}
				out = tenncor::if_then_else(condition, then, otherwise);
			}
				break;
			case egen::ARGMAX:
				logs::fatalf("cannot derive %s", opcode.name_.c_str());
				break;
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
		return (to_link<T>(lhs) + to_link<T>(rhs))->get_tensor();
	}
};

/// Derive root with respect to target and optimized
template <typename T>
LinkptrT<T> derive (LinkptrT<T> root, LinkptrT<T> target)
{
	GradientBuilder<T> builder;
	teq::TensptrT derivative = builder.derive(
		root->get_tensor(), target->get_tensor());
	return to_link<T>(derivative);
}

}

#endif // ETEQ_GRADER_HPP
