//
/// funcarg.hpp
/// eteq
///
/// Purpose:
/// Typed Eigen node version of teq::FuncArg
///

#include "teq/funcarg.hpp"

#include "eigen/coord.hpp"

#include "eteq/inode.hpp"

#ifndef ETEQ_FUNCARG_HPP
#define ETEQ_FUNCARG_HPP

namespace eteq
{

/// Eigen node version of teq::FuncArg
template <typename T>
struct FuncArg final
{
	/// Construct FuncArg with specific node, shaper, and coorder
	FuncArg (NodeptrT<T> node, teq::CoordptrT shaper, eigen::CoordptrT coorder) :
		node_(node), shaper_(shaper), coorder_(coorder)
	{
		if (node_ == nullptr)
		{
			logs::fatal("cannot map a null node");
		}
	}

	/// Return shape of tensor filtered through coordinate mapper
	teq::Shape shape (void) const
	{
		return teq::apply_shaper(shaper_, node_->shape());
	}

	/// Return tensor being mapped
	teq::TensptrT get_tensor (void) const
	{
		return node_->get_tensor();
	}

	NodeptrT<T> get_node (void) const
	{
		return node_;
	}

	/// Return shaper coord map
	teq::CoordptrT get_shaper (void) const
	{
		return shaper_;
	}

	/// Return map_io_ flag, True if coorder accepts input coord
	/// and generated output, False otherwise
	bool map_io (void) const
	{
		// map in to out only if bijective
		return nullptr == coorder_ || coorder_->is_bijective();
	}

	/// Return coord map for coordinates
	eigen::CoordptrT get_coorder (void) const
	{
		return coorder_;
	}

private:
	/// Tensor reference
	NodeptrT<T> node_;

	/// Shape mapper
	teq::CoordptrT shaper_;

	/// Coordinate mapper
	eigen::CoordptrT coorder_;
};

/// Type of typed functor arguments
template <typename T>
using ArgsT = std::vector<FuncArg<T>>;

template <typename T>
using PairVecT = std::vector<std::pair<T,T>>;

/// Return FuncArg<T> that identity maps input tensor
template <typename T>
FuncArg<T> identity_map (NodeptrT<T> node)
{
	return FuncArg<T>(node, teq::identity, nullptr);
}

/// Return FuncArg<T> that reduces input tensor by
/// units in reduction vector after specified rank
/// E.g.: tensor w/ shape [2, 3, 4], offset = 1, ndims = 2
/// gets mapped to [2, 1, 1]
template <typename T>
FuncArg<T> reduce_map (NodeptrT<T> node, teq::RankT offset, teq::RankT ndims)
{
	if (offset >= teq::rank_cap)
	{
		logs::fatalf("cannot reduce dimensions [%d:]. Must be less than %d",
			offset, teq::rank_cap);
	}

	teq::RankT n = std::min<teq::RankT>(offset + ndims, teq::rank_cap);
	teq::Shape shape = node->shape();
	std::vector<teq::RankT> dims; // dims are allowed to be non-contiguous
	std::vector<teq::DimT> slist;
	dims.reserve(n);
	slist.reserve(n);

	for (teq::RankT i = offset; i < n; ++i)
	{
		if (shape.at(i) > 1)
		{
			dims.push_back(i);
		}
		slist.push_back(shape.at(i));
	}

	return FuncArg<T>(node, teq::reduce(offset, slist), eigen::reduce(dims));
}

/// Return FuncArg<T> that reduce tensor by argument index
/// return_dim greater than rank_cap looks across all dimensions
template <typename T>
FuncArg<T> argreduce_map (NodeptrT<T> node, teq::RankT return_dim)
{
	teq::Shape shape = node->shape();
	teq::RankT offset;
	std::vector<teq::DimT> slist;
	if (return_dim >= teq::rank_cap)
	{
		offset = 0;
		slist = std::vector<teq::DimT>(shape.begin(), shape.end());
	}
	else
	{
		offset = return_dim;
		slist = {shape.at(offset)};
	}

	return FuncArg<T>(node, teq::reduce(offset, slist),
		std::make_shared<eigen::CoordMap>(
			[&](teq::MatrixT& args)
			{
				args[0][0] = return_dim;
			}));
}

/// Return FuncArg<T> that extends input tensor by
/// rank and extension vector
/// E.g.: tensor w/ shape [2, 1, 1], rank = 1, ext = [3, 4]
/// gets mapped to [2, 3, 4]
template <typename T>
FuncArg<T> extend_map (NodeptrT<T> node,
	teq::RankT rank, std::vector<teq::DimT> ext)
{
	if (0 == ext.size())
	{
		logs::fatal("cannot extend with empty vector");
	}
	return FuncArg<T>(node, teq::extend(rank, ext), eigen::extend(rank, ext));
}

/// Return FuncArg<T> that permutes input tensor by order
/// E.g.: tensor w/ shape [2, 3, 4], order = [1, 2, 0]
/// gets mapped to [3, 4, 2]
template <typename T>
FuncArg<T> permute_map (NodeptrT<T> node, std::vector<teq::RankT> order)
{
	return FuncArg<T>(node, teq::permute(order), eigen::permute(order));
}

/// Return FuncArg<T> that reshapes node to specified shape
template <typename T>
FuncArg<T> reshape_map (NodeptrT<T> node, const teq::Shape& shape)
{
	return FuncArg<T>(node,
		std::make_shared<teq::CoordMap>(
			[=](teq::MatrixT& fwd)
			{
				for (teq::RankT i = 0; i < teq::rank_cap; ++i)
				{
					fwd[teq::rank_cap][i] = shape.at(i);
				}
			}), nullptr);
}

/// Return FuncArg<T> that takes specific slice of tensor according to
/// vector of offset, extent pairs
/// E.g.: tensor w/ shape [2, 3, 4], offset = 1, extent = 2, and dimension = 2
/// gets mapped to [3, 4, 2] that references [:,:,1:3]
/// (second and third slices of the 3rd dimension)
template <typename T>
FuncArg<T> slice_map (NodeptrT<T> node, const PairVecT<teq::DimT>& extents)
{
	if (extents.size() > teq::rank_cap)
	{
		PairVecT<int> readable_extents(extents.begin(), extents.end());
		logs::fatalf(
			"cannot slice dimensions beyond rank_cap %d: using extent %s",
			teq::rank_cap,
			fmts::to_string(readable_extents.begin(), readable_extents.end()).c_str());
	}
	teq::Shape shape = node->shape();
	teq::CoordT offsets;
	teq::CoordT ns;
	teq::RankT n = extents.size();
	for (teq::RankT i = 0; i < n; ++i)
	{
		auto& ex = extents[i];
		if (ex.second < 1)
		{
			PairVecT<int> readable_extents(extents.begin(), extents.end());
			logs::fatalf("cannot extend zero slices: extents %s",
				fmts::to_string(readable_extents.begin(), readable_extents.end()).c_str());
		}
		teq::DimT offset = std::min(ex.first, (teq::DimT) (shape.at(i) - 1));
		offsets[i] = offset;
		ns[i] = std::min(ex.second, (teq::DimT) (shape.at(i) - offset));
	}
	for (teq::RankT i = n; i < teq::rank_cap; ++i)
	{
		ns[i] = shape.at(i);
	}
	return FuncArg<T>(node,
		std::make_shared<teq::CoordMap>(
			[=](teq::MatrixT& fwd)
			{
				for (teq::RankT i = 0; i < teq::rank_cap; ++i)
				{
					fwd[i][i] = 1;
					fwd[teq::rank_cap][i] = ns[i] - shape.at(i);
				}
			}),
		std::make_shared<eigen::CoordMap>(
			[&](teq::MatrixT& args)
			{
				args[0][teq::rank_cap] = 0; // mark contiguous zones as non-nan
				for (size_t i = 0; i < teq::rank_cap; ++i)
				{
					args[0][i] = offsets[i];
					args[1][i] = ns[i];
				}
			}));
}

/// Return FuncArg<T> that pads tensor with 0s across specified dimensions
/// E.g.: tensor w/ shape [2, 3, 4], padding = {2,1}, dimension = 0
/// gets mapped to [5, 3, 4] where [0,:,:] and [3:5,:,:] are 0
/// (first, fourth, and fifth slices of the 1st dimension are 0)
template <typename T>
FuncArg<T> pad_map (NodeptrT<T> node, const PairVecT<teq::DimT>& paddings)
{
	if (paddings.size() > teq::rank_cap)
	{
		PairVecT<int> readable_paddings(paddings.begin(), paddings.end());
		logs::fatalf(
			"cannot pad dimensions beyond rank_cap %d: using paddings %s",
			teq::rank_cap,
			fmts::to_string(readable_paddings.begin(), readable_paddings.end()).c_str());
	}
	return FuncArg<T>(node,
		std::make_shared<teq::CoordMap>(
			[=](teq::MatrixT& fwd)
			{
				teq::RankT n = std::min((teq::RankT) paddings.size(), teq::rank_cap);
				for (teq::RankT i = 0; i < n; ++i)
				{
					fwd[i][i] = 1;
					fwd[teq::rank_cap][i] =
						paddings[i].first + paddings[i].second;
				}
				for (teq::RankT i = n; i < teq::rank_cap; ++i)
				{
					fwd[i][i] = 1;
				}
			}),
		std::make_shared<eigen::CoordMap>(
			[&](teq::MatrixT& args)
			{
				args[0][teq::rank_cap] = 0; // mark contiguous zones as non-nan
				teq::RankT n = std::min((teq::RankT) paddings.size(), teq::rank_cap);
				for (size_t i = 0; i < n; ++i)
				{
					args[0][i] = paddings[i].first;
					args[1][i] = paddings[i].second;
				}
				for (size_t i = n; i < teq::rank_cap; ++i)
				{
					args[0][i] = args[1][i] = 0;
				}
			}));
}

/// Return FuncArg<T> that takes elements of
/// specific increments across dimensions starting from 0
/// E.g.: tensor w/ shape [2, 3, 4], incrs = {1, 2, 2}
/// gets mapped to [2, 2, 2] where
/// output[:,0,0] takes on input[:,0,0]
/// output[:,1,0] takes on input[:,2,0]
/// output[:,0,1] takes on input[:,0,2]
/// output[:,1,1] takes on input[:,2,2]
template <typename T>
FuncArg<T> stride_map (NodeptrT<T> node,
	const std::vector<teq::DimT>& incrs)
{
	if (incrs.size() > teq::rank_cap)
	{
		logs::warnf("trying to stride in dimensions beyond rank_cap %d: "
			"using increments %s (will ignore those dimensions)", teq::rank_cap,
			fmts::to_string(incrs.begin(), incrs.end()).c_str());
	}
	return FuncArg<T>(node,
		std::make_shared<teq::CoordMap>(
			[=](teq::MatrixT& fwd)
			{
				teq::RankT n = std::min((teq::RankT) incrs.size(), teq::rank_cap);
				for (teq::RankT i = 0; i < n; ++i)
				{
					// ceil(in_dim / stride) =
					// round(in_dim / stride + 0.5 - <smol num>)
					fwd[i][i] = 1. / incrs[i];
					fwd[teq::rank_cap][i] = 0.5 -
						1. / std::numeric_limits<teq::DimT>::max();
				}
				for (teq::RankT i = n; i < teq::rank_cap; ++i)
				{
					fwd[i][i] = 1;
				}
			}),
		std::make_shared<eigen::CoordMap>(
			[&](teq::MatrixT& args)
			{
				teq::RankT n = std::min((teq::RankT) incrs.size(), teq::rank_cap);
				for (size_t i = 0; i < n; ++i)
				{
					args[0][i] = incrs[i];
				}
				for (teq::RankT i = n; i < teq::rank_cap; ++i)
				{
					args[0][i] = 1;
				}
			}));
}

template <typename T>
ArgsT<T> convolve_map (NodeptrT<T> image, NodeptrT<T> kernel,
	const std::vector<teq::RankT>& dims)
{
	teq::Shape inshape = image->get_tensor()->shape();
	teq::Shape kernelshape = kernel->get_tensor()->shape();
	teq::CoordptrT input_shaper(new teq::CoordMap(
		[kernelshape,dims](teq::MatrixT& fwd)
		{
			for (teq::RankT i = 0; i < teq::rank_cap; ++i)
			{
				fwd[i][i] = 1;
			}
			teq::RankT n = std::min((teq::RankT) dims.size(), teq::rank_cap);
			for (teq::RankT i = 0; i < n; ++i)
			{
				fwd[teq::rank_cap][dims[i]] = -kernelshape.at(i) + 1;
			}
			if (std::any_of(kernelshape.begin() + n, kernelshape.end(),
				[](teq::DimT d)
				{
					return d > 1;
				}))
			{
				logs::fatalf("invalid kernelshape %s does not solely match dimensions %s",
					kernelshape.to_string().c_str(),
					fmts::to_string(dims.begin(), dims.end()).c_str());
			}
		}
	));

	teq::CoordptrT kernel_shaper(new teq::CoordMap(
		[inshape,dims](teq::MatrixT& fwd)
		{
			teq::RankT n = std::min((teq::RankT) dims.size(), teq::rank_cap);
			std::array<bool,teq::rank_cap> missing;
			std::fill(missing.begin(), missing.end(), true);
			for (teq::RankT i = 0; i < n; ++i)
			{
				missing[dims[i]] = false;
			}
			std::vector<teq::RankT> refd = dims;
			for (teq::RankT i = 0; i < teq::rank_cap; ++i)
			{
				if (missing[i])
				{
					refd.push_back(i);
				}
			}
			for (teq::RankT i = 0; i < teq::rank_cap; ++i)
			{
				fwd[i][refd[i]] = -1;
				fwd[teq::rank_cap][refd[i]] = inshape.at(refd[i]) + 1;
			}
		}
	));

	teq::CoordT kernel_dims;
	auto it = kernel_dims.begin();
	std::fill(it, kernel_dims.end(), teq::rank_cap);
	std::copy(dims.begin(), dims.end(), it);
	return {
		FuncArg<T>(image, input_shaper, nullptr),
		FuncArg<T>(kernel, kernel_shaper,
			std::make_shared<eigen::CoordMap>(kernel_dims)),
	};
}

template <typename T>
ArgsT<T> concat_map (NodeptrT<T> left, NodeptrT<T> right, teq::RankT axis)
{
	teq::Shape leftshape = left->shape();
	teq::Shape rightshape = right->shape();
	teq::CoordptrT left_shaper(new teq::CoordMap(
		[axis, rightshape](teq::MatrixT& fwd)
		{
			for (teq::RankT i = 0; i < teq::rank_cap; ++i)
			{
				fwd[i][i] = 1;
			}
			fwd[teq::rank_cap][axis] = rightshape.at(axis);
		}
	));
	teq::CoordptrT right_shaper(new teq::CoordMap(
		[axis, leftshape](teq::MatrixT& fwd)
		{
			for (teq::RankT i = 0; i < teq::rank_cap; ++i)
			{
				fwd[i][i] = 1;
			}
			fwd[teq::rank_cap][axis] = leftshape.at(axis);
		}
	));

	return {
		FuncArg<T>(left, left_shaper, std::make_shared<eigen::CoordMap>(
			[axis](teq::MatrixT& args)
			{
				args[0][0] = axis;
			})),
		FuncArg<T>(right, right_shaper, nullptr),
	};
}

template <typename T>
ArgsT<T> group_concat_map (NodesT<T> args, teq::RankT axis)
{
	if (args.size() < 2)
	{
		logs::fatal("cannot group concat less than 2 arguments");
	}
	if (std::any_of(args.begin(), args.end(),
		[](NodeptrT<T> arg) { return nullptr == arg; }))
	{
		logs::fatal("cannot group concat with null argument");
	}
	teq::DimT nargs = args.size();
	teq::CoordptrT shaper(new teq::CoordMap(
		[axis, nargs](teq::MatrixT& fwd)
		{
			for (teq::RankT i = 0; i < teq::rank_cap; ++i)
			{
				fwd[i][i] = 1;
			}
			fwd[teq::rank_cap][axis] = nargs - 1;
		}
	));
	ArgsT<T> out;
	out.reserve(nargs);
	out.push_back(FuncArg<T>(args[0], shaper, std::make_shared<eigen::CoordMap>(
		[axis](teq::MatrixT& args)
		{
			args[0][0] = axis;
		})));
	std::transform(args.begin() + 1, args.end(), std::back_inserter(out),
		[&](NodeptrT<T> arg)
		{
			return FuncArg<T>(arg, shaper, nullptr);
		});
	return out;
}

template <typename T>
ArgsT<T> contract_map (NodeptrT<T> a, NodeptrT<T> b, PairVecT<teq::RankT> dims)
{
	teq::Shape ashape = a->get_tensor()->shape();
	teq::Shape bshape = b->get_tensor()->shape();
	// check common dimensions
	std::array<bool,teq::rank_cap> avisit;
	std::array<bool,teq::rank_cap> bvisit;
	std::fill(avisit.begin(), avisit.end(), false);
	std::fill(bvisit.begin(), bvisit.end(), false);
	for (std::pair<teq::RankT,teq::RankT>& coms : dims)
	{
		if (ashape.at(coms.first) != bshape.at(coms.second))
		{
			PairVecT<int> readable_dims(dims.begin(), dims.end());
			logs::fatalf("invalid shapes %s and %s do not match common dimensions %s",
				ashape.to_string().c_str(), bshape.to_string().c_str(),
				fmts::to_string(readable_dims.begin(), readable_dims.end()).c_str());
		}
		if (avisit[coms.first] || bvisit[coms.second])
		{
			PairVecT<int> readable_dims(dims.begin(), dims.end());
			logs::fatalf("contraction dimensions %s must be unique for each side",
				fmts::to_string(readable_dims.begin(), readable_dims.end()).c_str());
		}
		avisit[coms.first] = bvisit[coms.second] = true;
	}
	std::vector<teq::DimT> alist = teq::narrow_shape(ashape);
	std::vector<teq::DimT> blist = teq::narrow_shape(bshape);
	std::vector<teq::DimT> outlist;
	outlist.reserve(teq::rank_cap);
	for (teq::RankT i = 0, n = blist.size(); i < n; ++i)
	{
		if (false == bvisit[i])
		{
			outlist.push_back(blist.at(i));
		}
	}
	for (teq::RankT i = 0, n = alist.size(); i < n; ++i)
	{
		if (false == avisit[i])
		{
			outlist.push_back(alist.at(i));
		}
	}
	if (teq::rank_cap > outlist.size())
	{
		outlist.insert(outlist.end(), teq::rank_cap - outlist.size(), 1);
	}

	teq::CoordptrT left_shaper(new teq::CoordMap(
		[&](teq::MatrixT& fwd)
		{
			for (teq::RankT i = 0; i < teq::rank_cap; ++i)
			{
				fwd[i][i] = outlist[i] != ashape.at(i) ?
					(double) outlist[i] / ashape.at(i) : 1.;
			}
		}
	));

	teq::CoordptrT right_shaper(new teq::CoordMap(
		[&](teq::MatrixT& fwd)
		{
			for (teq::RankT i = 0; i < teq::rank_cap; ++i)
			{
				fwd[i][i] = outlist[i] != bshape.at(i) ?
					(double) outlist[i] / bshape.at(i) : 1;
			}
		}
	));

	return {
		eteq::FuncArg<T>(a, left_shaper, std::make_shared<eigen::CoordMap>(
			[&dims](teq::MatrixT& args)
			{
				args[0][teq::rank_cap] = 0; // mark contiguous zones as non-nan
				teq::RankT n = std::min((teq::RankT) dims.size(), teq::rank_cap);
				for (size_t i = 0; i < n; ++i)
				{
					args[0][i] = dims[i].first;
					args[1][i] = dims[i].second;
				}
				for (size_t i = n; i < teq::rank_cap; ++i)
				{
					args[0][i] = args[1][i] = teq::rank_cap;
				}
			})),
		eteq::FuncArg<T>(b, right_shaper, nullptr),
	};
}

}

#endif // ETEQ_FUNCARG_HPP
