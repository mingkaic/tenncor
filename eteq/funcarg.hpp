//
/// funcarg.hpp
/// eteq
///
/// Purpose:
/// Typed Eigen node version of teq::FuncArg
///

#include "teq/funcarg.hpp"

#include "eteq/coord.hpp"
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
	FuncArg (NodeptrT<T> node, teq::CoordptrT shaper, CoordptrT coorder) :
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
	CoordptrT get_coorder (void) const
	{
		return coorder_;
	}

private:
	/// Tensor reference
	NodeptrT<T> node_;

	/// Shape mapper
	teq::CoordptrT shaper_;

	/// Coordinate mapper
	CoordptrT coorder_;
};

/// Type of typed functor arguments
template <typename T>
using ArgsT = std::vector<FuncArg<T>>;

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

	return FuncArg<T>(node, teq::reduce(offset, slist), reduce(dims));
}

/// Return FuncArg<T> that extends input tensor by
/// rank and extension vector
/// E.g.: tensor w/ shape [2, 1, 1], rank = 1, ext = [3, 4]
/// gets mapped to [2, 3, 4]
template <typename T>
FuncArg<T> extend_map (NodeptrT<T> node,
	teq::RankT rank, std::vector<teq::DimT> ext)
{
	return FuncArg<T>(node, teq::extend(rank, ext), extend(rank, ext));
}

/// Return FuncArg<T> that permutes input tensor by order
/// E.g.: tensor w/ shape [2, 3, 4], order = [1, 2, 0]
/// gets mapped to [3, 4, 2]
template <typename T>
FuncArg<T> permute_map (NodeptrT<T> node, std::vector<teq::RankT> order)
{
	return FuncArg<T>(node, teq::permute(order), permute(order));
}

template <typename T>
using PairVecT = std::vector<std::pair<T,T>>;

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
		logs::fatalf(
			"cannot slice dimensions beyond rank_cap %d: using extent %s",
			teq::rank_cap,
			fmts::to_string(extents.begin(), extents.end()).c_str());
	}
	teq::Shape shape = node->shape();
	teq::CoordT offsets;
	teq::CoordT ns;
	for (size_t i = 0, n = extents.size(); i < n; ++i)
	{
		auto& ex = extents[i];
		teq::DimT offset = std::min(ex.first, shape.at(i));
		offsets[i] = offset;
		ns[i] = std::min(ex.second, (teq::DimT) (shape.at(i) - offset));
	}
	return FuncArg<T>(node,
		std::make_shared<teq::CoordMap>(
			[=](teq::MatrixT fwd)
			{
				for (teq::RankT i = 0; i < teq::rank_cap; ++i)
				{
					fwd[i][i] = 1;
					fwd[teq::rank_cap][i] = ns[i] - shape.at(i);
				}
			}),
		std::make_shared<CoordMap>(
			[&](teq::MatrixT args)
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
		logs::fatalf(
			"cannot pad dimensions beyond rank_cap %d: using paddings %s",
			teq::rank_cap,
			fmts::to_string(paddings.begin(), paddings.end()).c_str());
	}
	return FuncArg<T>(node,
		std::make_shared<teq::CoordMap>(
			[=](teq::MatrixT fwd)
			{
				for (teq::RankT i = 0; i < teq::rank_cap; ++i)
				{
					fwd[i][i] = 1;
					fwd[teq::rank_cap][i] =
						paddings[i].first + paddings[i].second;
				}
			}),
		std::make_shared<CoordMap>(
			[&](teq::MatrixT args)
			{
				args[0][teq::rank_cap] = 0; // mark contiguous zones as non-nan
				for (size_t i = 0; i < teq::rank_cap; ++i)
				{
					args[0][i] = paddings[i].first;
					args[1][i] = paddings[i].second;
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
			[=](teq::MatrixT fwd)
			{
				for (teq::RankT i = 0; i < teq::rank_cap; ++i)
				{
					// ceil(in_dim / stride) =
					// round(in_dim / stride + 0.5 - <smol num>)
					fwd[i][i] = 1. / incrs[i];
					fwd[teq::rank_cap][i] = 0.5 -
						1. / std::numeric_limits<teq::DimT>::max();
				}
			}),
		std::make_shared<CoordMap>(
			[&](teq::MatrixT args)
			{
				for (size_t i = 0; i < teq::rank_cap; ++i)
				{
					args[0][i] = incrs[i];
				}
			}));
}

}

#endif // ETEQ_FUNCARG_HPP
