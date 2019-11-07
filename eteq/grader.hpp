///
/// grader.hpp
/// eteq
///
/// Purpose:
/// Implement eteq gradient definition for supported operations
///

#include "teq/grad_def.hpp"

#include "eteq/generated/api.hpp"
#include "eteq/constant.hpp"

#ifndef ETEQ_GRADER_HPP
#define ETEQ_GRADER_HPP

namespace eteq
{

/// Return reduction operator gradient of reduced functor node (bwd)
template <typename T>
NodeptrT<T> reduce_grad (const teq::iFuncArg& child,
	NodeptrT<T> bwd, size_t idx)
{
	const teq::Shape& shape = child.argshape();
	teq::CoordptrT revshaper(child.get_shaper()->reverse());
	eigen::CoordptrT revcoord;
	{
		auto coorder = static_cast<const teq::FuncArg*>(&child)->get_coorder();
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
		revcoord = std::make_shared<eigen::CoordMap>(bcast, false);
	}
	return make_functor<T>(teq::Opcode{"EXTEND",egen::EXTEND}, {
		FuncArg<T>(bwd, revshaper, revcoord)
	});
}

/// Return permutation gradient of permuted functor node (bwd)
template <typename T>
NodeptrT<T> permute_grad (teq::iFunctor* fwd,
	NodeptrT<T> bwd, size_t idx) // move this below
{
	const teq::iFuncArg& child = fwd->get_children()[0];
	teq::CoordptrT revshaper(child.get_shaper()->reverse());
	eigen::CoordptrT revcoord;
	{
		auto coorder = static_cast<const teq::FuncArg*>(&child)->get_coorder();
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
		revcoord = std::make_shared<eigen::CoordMap>(order, true);
	}
	return make_functor<T>(teq::Opcode{"PERMUTE",egen::PERMUTE},{
		FuncArg<T>(bwd, revshaper, revcoord)
	});
}

/// Return extension gradient of extended functor node (bwd)
template <typename T>
NodeptrT<T> extend_grad (teq::iFunctor* fwd,
	NodeptrT<T> bwd, size_t idx) // move this below
{
	const teq::iFuncArg& child = fwd->get_children()[0];
	teq::CoordptrT revshaper(child.get_shaper()->reverse());
	eigen::CoordptrT revcoord;
	{
		auto coorder = static_cast<const teq::FuncArg*>(&child)->get_coorder();
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
		revcoord = eigen::reduce(red_dims);
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
		auto args = op->get_children();
		NodeptrT<T> out = nullptr;
		teq::Opcode opcode = op->get_opcode();
		switch ((egen::_GENERATED_OPCODE) opcode.code_)
		{
			case egen::ABS:
				out = to_node<T>(args[0].get().get_tensor()) / to_node<T>(op);
				break;
			case egen::NEG:
				out = make_constant_scalar<T>(
					-1, args[0].get().argshape());
				break;
			case egen::SIN:
				out = tenncor::cos(to_node<T>(args[0].get().get_tensor()));
				break;
			case egen::COS:
				out = -tenncor::sin(to_node<T>(args[0].get().get_tensor()));
				break;
			case egen::TAN:
				out = (T) 1 / tenncor::pow(
					tenncor::cos(to_node<T>(args[0].get().get_tensor())), (T) 2);
				break;
			case egen::EXP:
				out = to_node<T>(op);
				break;
			case egen::LOG:
				out = (T) 1 / to_node<T>(args[0].get().get_tensor());
				break;
			case egen::SQRT:
				out = (T) 1 / ((T) 2 * to_node<T>(op));
				break;
			case egen::SQUARE:
				out = (T) 2 * to_node<T>(args[0].get().get_tensor());
				break;
			case egen::CUBE:
				out = (T) 3 * tenncor::square(to_node<T>(args[0].get().get_tensor()));
				break;
			case egen::SIGMOID:
				out = to_node<T>(op) * ((T) 1 - to_node<T>(op));
				break;
			case egen::TANH:
				out = (T) 1 - tenncor::square(to_node<T>(op));
				break;
			case egen::ROUND:
			case egen::REDUCE_SUM:
			case egen::EXTEND:
			case egen::PERMUTE:
			case egen::RESHAPE:
			case egen::ADD:
			case egen::SLICE:
			case egen::PAD:
			case egen::STRIDE:
			case egen::SCATTER:
			case egen::REVERSE:
			case egen::CONV:
			case egen::CONCAT:
			case egen::GROUP_CONCAT:
				out = make_constant_scalar<T>(1, args[arg_idx].get().argshape());
				break;
			case egen::MUL:
				out = to_node<T>(args[(size_t)(arg_idx==0)].get().get_tensor());
				break;
			case egen::MAX:
			case egen::MIN:
				out = to_node<T>(op) == to_node<T>(args[arg_idx].get().get_tensor());
				break;
			case egen::POW:
				out = arg_idx==0 ?
					to_node<T>(args[1].get().get_tensor()) *
					tenncor::pow(
						to_node<T>(args[0].get().get_tensor()),
						to_node<T>(args[1].get().get_tensor()) - (T) 1
					) :
					tenncor::log(to_node<T>(args[0].get().get_tensor())) *
						to_node<T>(op);
				break;
			case egen::SUB:
				out = make_constant_scalar<T>(arg_idx == 0 ?
					1 : -1, args[0].get().argshape());
				break;
			case egen::DIV:
			{
				auto denom = to_node<T>(args[1].get().get_tensor());
				out = arg_idx==0 ?
					(T) 1 / denom :
					-to_node<T>(args[0].get().get_tensor()) / denom / denom;
			}
				break;
			case egen::EQ:
			case egen::NEQ:
			case egen::GT:
			case egen::LT:
			case egen::RAND_UNIF:
			case egen::SELECT:
				out = make_constant_scalar<T>(0, args[0].get().argshape());
				break;
			case egen::REDUCE_PROD: // todo: prevent divide by zero
				out =
					reduce_grad(args[0], to_node<T>(op), arg_idx) /
					to_node<T>(args[0].get().get_tensor());
				break;
			case egen::REDUCE_MAX:
			case egen::REDUCE_MIN:
				out =
					reduce_grad(args[0], to_node<T>(op), arg_idx) ==
					to_node<T>(args[0].get().get_tensor());
				break;
			case egen::MATMUL:
			{
				auto coorder = static_cast<const teq::FuncArg*>(&args[0].get())->get_coorder();
				NodeptrT<T> lhs = to_node<T>(args[0].get().get_tensor());
				NodeptrT<T> rhs = to_node<T>(args[1].get().get_tensor());
				assert(nullptr != coorder);
				PairVecT<teq::RankT> dims;
				coorder->access(
					[&](const teq::MatrixT& args)
					{
						for (teq::RankT i = 0; i < teq::rank_cap &&
							args[0][i] < teq::rank_cap; ++i)
						{
							dims.push_back({args[0][i], args[1][i]});
						}
					});
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
				NodeptrT<T> ext;
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
				out = to_node<T>(local_der) *
					to_node<T>(supcomp_grad);
				break;
			case egen::REDUCE_MAX:
			case egen::REDUCE_MIN:
			case egen::REDUCE_PROD:
			case egen::REDUCE_SUM:
				out = to_node<T>(local_der) * reduce_grad(
					op->get_children()[0], to_node<T>(supcomp_grad), arg_idx);
				break;
			case egen::EXTEND:
				out = to_node<T>(local_der) * extend_grad(
					op.get(), to_node<T>(supcomp_grad), arg_idx);
				break;
			case egen::PERMUTE:
				out = to_node<T>(local_der) * permute_grad(
					op.get(), to_node<T>(supcomp_grad), arg_idx);
				break;
			case egen::RESHAPE:
			{
				const teq::iFuncArg& child = op->get_children()[0];
				out = to_node<T>(local_der) * tenncor::reshape(
					to_node<T>(supcomp_grad), child.argshape());
			}
				break;
			case egen::MATMUL:
			{
				auto args = op->get_children();
				auto coorder = static_cast<const teq::FuncArg*>(&args[0].get())->get_coorder();
				assert(nullptr != coorder);
				PairVecT<teq::RankT> dims;
				coorder->access(
					[&](const teq::MatrixT& args)
					{
						for (teq::RankT i = 0; i < teq::rank_cap &&
							args[0][i] < teq::rank_cap; ++i)
						{
							dims.push_back({args[0][i], args[1][i]});
						}
					});

				std::vector<teq::DimT> llist = teq::narrow_shape(args[0].get().argshape());
				std::vector<teq::DimT> rlist = teq::narrow_shape(args[1].get().argshape());
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
				auto ext = tenncor::extend(to_node<T>(supcomp_grad),
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
				teq::RankT nperms = fwdperm.size();
				std::vector<teq::RankT> permlist(nperms);
				for (teq::RankT i = 0; i < nperms; ++i)
				{
					permlist[fwdperm[i]] = i;
				}

				// reduce <uncommon not arg_idx> such that shape = <arg_idx shape>
				out = tenncor::reduce_sum(
					tenncor::permute(to_node<T>(local_der) * ext, permlist),
					arg_rank, teq::rank_cap - arg_rank);
			}
				break;
			case egen::CONV:
			{
				// for convolution(X, Y) = C
				auto args = op->get_children();
				std::vector<teq::RankT> dims;
				auto coorder = static_cast<const teq::FuncArg*>(&args[1].get())->get_coorder();
				assert(nullptr != coorder);
				coorder->access(
					[&](const teq::MatrixT& args)
					{
						for (teq::RankT i = 0; i < teq::rank_cap &&
							args[0][i] < teq::rank_cap; ++i)
						{
							dims.push_back(args[0][i]);
						}
					});
				if (arg_idx == 0)
				{
					// convolve(pad(C_grad_sup, Y.shape[dims]-1), reverse(Y))
					teq::RankT ndims = dims.size();
					teq::Shape kernshape = args[1].get().argshape();
					PairVecT<teq::DimT> paddings(teq::rank_cap, {0, 0});
					for (teq::RankT i = 0; i < ndims; ++i)
					{
						teq::DimT kpad = kernshape.at(i) - 1;
						paddings[dims[i]] = {kpad, kpad};
					}
					std::vector<teq::RankT> revdims(ndims);
					std::iota(revdims.begin(), revdims.end(), 0);
					out = tenncor::convolution(tenncor::pad(
						to_node<T>(supcomp_grad), paddings),
						tenncor::reverse(
							to_node<T>(args[1].get().get_tensor()), revdims), dims);
				}
				else
				{
					// convolve(X, C_grad_sup)
					std::vector<teq::RankT> indices(teq::rank_cap);
					std::iota(indices.begin(), indices.end(), 0);
					out = tenncor::permute(
						tenncor::convolution(
							to_node<T>(args[0].get().get_tensor()),
							to_node<T>(supcomp_grad),
							indices), dims);
				}
			}
				break;
			case egen::SLICE:
			{
				const teq::iFuncArg& child = op->get_children()[0];
				teq::ShapeT offsets;
				teq::ShapeT extents;
				static_cast<const teq::FuncArg*>(&child)->get_coorder()->access(
					[&](const teq::MatrixT& args)
					{
						std::copy(args[0], args[0] + teq::rank_cap, offsets.begin());
						std::copy(args[1], args[1] + teq::rank_cap, extents.begin());
					});
				teq::Shape cshape = child.argshape();
				PairVecT<teq::DimT> paddings;
				paddings.reserve(teq::rank_cap);
				for (size_t i = 0; i < teq::rank_cap; ++i)
				{
					teq::DimT leftpad = offsets[i];
					paddings.push_back({leftpad,
						cshape.at(i) - (leftpad + extents[i])});
				}
				out = to_node<T>(local_der) *
					tenncor::pad(to_node<T>(supcomp_grad), paddings);
			}
				break;
			case egen::PAD:
			{
				const teq::iFuncArg& child = op->get_children()[0];
				teq::ShapeT leftpad;
				teq::ShapeT rightpad;
				static_cast<const teq::FuncArg*>(&child)->get_coorder()->access(
					[&](const teq::MatrixT& args)
					{
						std::copy(args[0], args[0] + teq::rank_cap,
							leftpad.begin());
						std::copy(args[1], args[1] + teq::rank_cap,
							rightpad.begin());
					});
				teq::Shape oshape = op->shape();
				PairVecT<teq::DimT> extents;
				extents.reserve(teq::rank_cap);
				for (size_t i = 0; i < teq::rank_cap; ++i)
				{
					teq::DimT offset = leftpad[i];
					extents.push_back({offset,
						oshape.at(i) - rightpad[i] - offset});
				}
				out = to_node<T>(local_der) *
					tenncor::slice(to_node<T>(supcomp_grad), extents);
			}
				break;
			case egen::CONCAT:
			{
				auto children = op->get_children();
				const teq::iFuncArg& fchild = children[0];
				teq::Shape cshape = children[arg_idx].get().argshape();
				auto coorder = static_cast<const teq::FuncArg*>(&fchild)->get_coorder();
				assert(nullptr != coorder);
				teq::RankT axis;
				coorder->access(
					[&](const teq::MatrixT& args)
					{
						axis = args[0][0];
					});
				teq::DimT offset = 0;
				teq::DimT extent = cshape.at(axis);
				if (arg_idx)
				{
					teq::Shape first_shape = fchild.argshape();

					offset = first_shape.at(axis);
				}
				out = to_node<T>(local_der) *
					tenncor::slice(to_node<T>(supcomp_grad), offset, extent, axis);
			}
				break;
			case egen::GROUP_CONCAT: // todo: combine concat and group_concat
			{
				const teq::iFuncArg& fchild = op->get_children()[0];
				auto coorder = static_cast<const teq::FuncArg*>(&fchild)->get_coorder();
				assert(nullptr != coorder);
				teq::RankT axis;
				coorder->access(
					[&](const teq::MatrixT& args)
					{
						axis = args[0][0];
					});
				out = to_node<T>(local_der) *
					tenncor::slice(to_node<T>(supcomp_grad), arg_idx, 1, axis);
			}
				break;
			case egen::STRIDE:
			{
				const teq::iFuncArg& child = op->get_children()[0];
				teq::CoordT strides;
				static_cast<const teq::FuncArg*>(&child)->get_coorder()->access(
					[&](const teq::MatrixT& args)
					{
						std::copy(args[0], args[0] + teq::rank_cap,
							strides.begin());
					});
				teq::Shape origshape = child.argshape();
				out = to_node<T>(local_der) *
					make_functor<T>(teq::Opcode{"SCATTER",::egen::SCATTER}, {
						FuncArg<T>(to_node<T>(supcomp_grad),
							std::make_shared<teq::CoordMap>(
								[origshape,strides](teq::MatrixT& fwd)
								{
									teq::RankT n = std::min(
										(teq::RankT) strides.size(), teq::rank_cap);
									for (teq::RankT i = 0; i < n; ++i)
									{
										if (strides[i] == 1)
										{
											fwd[i][i] = 1;
										}
										else
										{
											// shape can't be trivially reconstructed
											fwd[teq::rank_cap][i] = origshape.at(i);
										}
									}
									for (teq::RankT i = n; i < teq::rank_cap; ++i)
									{
										fwd[i][i] = 1;
									}
								}),
							std::make_shared<eigen::CoordMap>(
								[&](teq::MatrixT& args)
								{
									for (size_t i = 0; i < teq::rank_cap; ++i)
									{
										args[0][i] = strides[i];
									}
								})
						)
					});
			}
				break;
			case egen::SCATTER:
			{
				const teq::iFuncArg& child = op->get_children()[0];
				std::vector<teq::DimT> strides;
				strides.reserve(teq::rank_cap);
				static_cast<const teq::FuncArg*>(&child)->get_coorder()->access(
					[&](const teq::MatrixT& args)
					{
						std::copy(args[0], args[0] + teq::rank_cap,
							std::back_inserter(strides));
					});
				out = to_node<T>(local_der) *
					tenncor::stride(to_node<T>(supcomp_grad), strides);
			}
				break;
			case egen::REVERSE:
				out = to_node<T>(local_der) * make_functor<T>(
					teq::Opcode{"REVERSE",egen::REVERSE}, {
						FuncArg<T>(
							to_node<T>(supcomp_grad),
							teq::identity,
							std::static_pointer_cast<eigen::CoordMap>(
								static_cast<const teq::FuncArg*>(&op->get_children()[0].get())->get_coorder())
						),
					});
				break;
			case egen::SELECT:
			{
				if (0 == arg_idx)
				{
					out = to_node<T>(local_der);
					break;
				}
				auto condition = to_node<T>(
					op->get_children()[0].get().get_tensor());
				auto then = to_node<T>(supcomp_grad);
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
		return teq::TensptrT(Functor<T>::get(teq::Opcode{"ADD", egen::ADD}, {
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
	teq::TensptrT derivative = builder.derive(
		root->get_tensor(), target->get_tensor());
	return to_node<T>(derivative);
}

}

#endif // ETEQ_GRADER_HPP
