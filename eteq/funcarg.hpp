//
/// funcarg.hpp
/// eteq
///
/// Purpose:
/// Typed Eigen implementation of teq iEdge
///

#include "eigen/edge.hpp"

#include "eteq/inode.hpp"

#ifndef ETEQ_FUNCARG_HPP
#define ETEQ_FUNCARG_HPP

namespace eteq
{

/// Implementation of iEigenEdge using node as tensor wrapper
template <typename T>
struct FuncArg final : public eigen::iEigenEdge<T>
{
	/// Construct FuncArg with specific node, shaper, and coorder
	FuncArg (NodeptrT<T> node, teq::ShaperT shaper,
		eigen::CoordptrT coorder) :
		node_(node), shaper_(shaper), coorder_(coorder)
	{
		if (node_ == nullptr)
		{
			logs::fatal("cannot map a null node");
		}
	}

	/// Implementation of iEdge
	teq::Shape shape (void) const override
	{
		teq::Shape out = this->argshape();
		if (nullptr != shaper_)
		{
			out = shaper_->convert(out);
		}
		return out;
	}

	/// Implementation of iEdge
	teq::Shape argshape (void) const override
	{
		return node_->shape();
	}

	/// Implementation of iEdge
	teq::TensptrT get_tensor (void) const override
	{
		return node_->get_tensor();
	}

	/// Implementation of iEdge
	void get_attrs (marsh::Maps& out) const
	{
		if (nullptr != shaper_)
		{
			auto arr = std::make_unique<marsh::NumArray<teq::CDimT>>();
			auto& contents = arr->contents_;
			shaper_->access(
				[&](const teq::MatrixT& args)
				{
					for (teq::RankT i = 0; i < teq::mat_dim; ++i)
					{
						for (teq::RankT j = 0; j < teq::mat_dim; ++j)
						{
							contents.push_back(args[i][j]);
						}
					}
				});
			out.contents_.emplace("shape", std::move(arr));
		}
		if (nullptr != coorder_)
		{
			auto arr = std::make_unique<marsh::NumArray<teq::CDimT>>();
			auto& contents = arr->contents_;
			coorder_->access(
				[&](const teq::MatrixT& args)
				{
					for (teq::RankT i = 0; i < teq::mat_dim &&
						false == std::isnan(args[i][0]); ++i)
					{
						for (teq::RankT j = 0; j < teq::mat_dim &&
						false == std::isnan(args[i][j]); ++j)
						{
							contents.push_back(args[i][j]);
						}
					}
				});
			out.contents_.emplace("coord", std::move(arr));
		}
	}

	/// Implementation of iEigenEdge<T>
	T* data (void) const override
	{
		return node_->data();
	}

	void set_tensor (teq::TensptrT tens)
	{
		node_ = to_node<T>(tens);
	}

	NodeptrT<T> get_node (void) const
	{
		return node_;
	}

private:
	/// Tensor reference
	NodeptrT<T> node_;

	/// Shape mapper
	teq::ShaperT shaper_;

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

	teq::Shape shape = node->shape();
	std::set<teq::RankT> dims; // dims are allowed to be non-contiguous
	for (size_t i = offset,
		n = std::min((size_t) offset + ndims, (size_t) teq::rank_cap);
		i < n; ++i)
	{
		if (shape.at(i) > 1)
		{
			dims.emplace(i);
		}
	}

	return FuncArg<T>(node, teq::reduce(dims), eigen::reduce(dims));
}

/// Return FuncArg<T> that reduce tensor by argument index
/// return_dim greater than rank_cap looks across all dimensions
template <typename T>
FuncArg<T> argreduce_map (NodeptrT<T> node, teq::RankT return_dim)
{
	std::set<teq::RankT> dims;
	if (return_dim >= teq::rank_cap)
	{
		std::array<teq::DimT,teq::rank_cap> indices;
		std::iota(indices.begin(), indices.end(), 0);
		dims.insert(indices.begin(), indices.end());
	}
	else
	{
		dims.emplace(return_dim);
	}

	return FuncArg<T>(node, teq::reduce(dims),
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
	size_t n_ext = ext.size();
	if (0 == n_ext)
	{
		logs::warn("extending with empty vector ... will do nothing");
		return identity_map(node);
	}
	if (std::any_of(ext.begin(), ext.end(),
		[](teq::DimT& d) { return 0 == d; }))
	{
		logs::fatalf("cannot extend using zero dimensions %s",
			fmts::to_string(ext.begin(), ext.end()).c_str());
	}
	if (rank + n_ext > teq::rank_cap)
	{
		logs::fatalf("cannot extend shape rank %d beyond rank_cap with n_ext %d",
			rank, n_ext);
	}
	teq::CoordT bcast;
	auto it = bcast.begin();
	std::fill(it, bcast.end(), 1);
	std::copy(ext.begin(), ext.end(), it + rank);
	return FuncArg<T>(node, teq::extend(bcast), eigen::extend(bcast));
}

/// Return FuncArg<T> that permutes input tensor by order
/// E.g.: tensor w/ shape [2, 3, 4], order = [1, 2, 0]
/// gets mapped to [3, 4, 2]
template <typename T>
FuncArg<T> permute_map (NodeptrT<T> node, std::vector<teq::RankT> order)
{
	if (order.size() == 0)
	{
		logs::warn("permuting with same dimensions ... will do nothing");
		return identity_map(node);
	}

	bool visited[teq::rank_cap];
	std::fill(visited, visited + teq::rank_cap, false);
	for (teq::RankT i = 0, n = order.size(); i < n; ++i)
	{
		if (visited[order[i]])
		{
			logs::fatalf("permute does not support repeated orders: %s",
				fmts::to_string(order.begin(), order.end()).c_str());
		}
		visited[order[i]] = true;
	}
	// since order can't be duplicate, norder < rank_cap
	for (teq::RankT i = 0; i < teq::rank_cap; ++i)
	{
		if (false == visited[i])
		{
			order.push_back(i);
		}
	}
	std::array<teq::RankT,teq::rank_cap> indices;
	std::copy(order.begin(), order.end(), indices.begin());
	return FuncArg<T>(node, teq::permute(indices), eigen::permute(indices));
}

/// Return FuncArg<T> that reshapes node to specified shape
template <typename T>
FuncArg<T> reshape_map (NodeptrT<T> node, const teq::Shape& shape)
{
	return FuncArg<T>(node,
		std::make_shared<teq::ShapeMap>(
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
	size_t n = extents.size();
	for (size_t i = 0; i < n; ++i)
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
		std::make_shared<teq::ShapeMap>(
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
		std::make_shared<teq::ShapeMap>(
			[=](teq::MatrixT& fwd)
			{
				size_t n = std::min(paddings.size(), (size_t) teq::rank_cap);
				for (size_t i = 0; i < n; ++i)
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
				size_t n = std::min(paddings.size(), (size_t) teq::rank_cap);
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
		std::make_shared<teq::ShapeMap>(
			[=](teq::MatrixT& fwd)
			{
				size_t n = std::min(incrs.size(), (size_t) teq::rank_cap);
				for (size_t i = 0; i < n; ++i)
				{
					fwd[i][i] = 1. / incrs[i];
				}
				for (teq::RankT i = n; i < teq::rank_cap; ++i)
				{
					fwd[i][i] = 1;
				}
			}),
		std::make_shared<eigen::CoordMap>(
			[&](teq::MatrixT& args)
			{
				size_t n = std::min(incrs.size(), (size_t) teq::rank_cap);
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
	teq::Shape inshape = image->shape();
	teq::Shape kernelshape = kernel->shape();
	teq::ShaperT input_shaper(new teq::ShapeMap(
		[kernelshape,dims](teq::MatrixT& fwd)
		{
			for (teq::RankT i = 0; i < teq::rank_cap; ++i)
			{
				fwd[i][i] = 1;
			}
			size_t n = std::min(dims.size(), (size_t) teq::rank_cap);
			for (size_t i = 0; i < n; ++i)
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

	teq::ShaperT kernel_shaper(new teq::ShapeMap(
		[inshape,dims](teq::MatrixT& fwd)
		{
			size_t n = std::min(dims.size(), (size_t) teq::rank_cap);
			std::array<bool,teq::rank_cap> missing;
			std::fill(missing.begin(), missing.end(), true);
			for (size_t i = 0; i < n; ++i)
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
	teq::ShaperT left_shaper(new teq::ShapeMap(
		[axis, rightshape](teq::MatrixT& fwd)
		{
			for (teq::RankT i = 0; i < teq::rank_cap; ++i)
			{
				fwd[i][i] = 1;
			}
			fwd[teq::rank_cap][axis] = rightshape.at(axis);
		}
	));
	teq::ShaperT right_shaper(new teq::ShapeMap(
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
	teq::ShaperT shaper(new teq::ShapeMap(
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
	teq::Shape ashape = a->shape();
	teq::Shape bshape = b->shape();
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

	teq::ShaperT left_shaper(new teq::ShapeMap(
		[&](teq::MatrixT& fwd)
		{
			for (teq::RankT i = 0; i < teq::rank_cap; ++i)
			{
				fwd[i][i] = outlist[i] != ashape.at(i) ?
					(double) outlist[i] / ashape.at(i) : 1.;
			}
		}
	));

	teq::ShaperT right_shaper(new teq::ShapeMap(
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
				size_t n = std::min(dims.size(), (size_t) teq::rank_cap);
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
