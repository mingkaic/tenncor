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

/// Return FuncArg<T> that takes specific slice of tensor
/// E.g.: tensor w/ shape [2, 3, 4], offset = 1, extent = 2, and dimension = 2
/// gets mapped to [3, 4, 2] that references [:,:,1:3]
/// (second and third slices of the 3rd dimension)
template <typename T>
FuncArg<T> slice_map (NodeptrT<T> node, teq::RankT offset,
	teq::RankT extent, teq::RankT dimension)
{
	if (dimension >= teq::rank_cap)
	{
		logs::fatalf("cannot slice dimension %d beyond rank_cap %d",
			dimension, teq::rank_cap);
	}
	teq::CoordT slicings;
	std::fill(slicings.begin(), slicings.end(), teq::rank_cap);
	slicings[0] = offset;
	slicings[1] = extent;
	slicings[2] = dimension;
	return FuncArg<T>(node,
		std::make_shared<teq::CoordMap>(
			[=](teq::MatrixT fwd)
			{
				for (teq::RankT i = 0; i < teq::rank_cap; ++i)
				{
					fwd[i][i] = 1;
				}
				fwd[teq::rank_cap][dimension] =
					extent - node->shape().at(dimension);
			}),
		std::make_shared<CoordMap>(slicings, false));
}

/// Return FuncArg<T> that pads tensor with 0s across specified dimension
/// E.g.: tensor w/ shape [2, 3, 4], padding = {2,1}, dimension = 0
/// gets mapped to [5, 3, 4] where [0,:,:] and [3:5,:,:] are 0
/// (first, fourth, and fifth slices of the 1st dimension are 0)
template <typename T>
FuncArg<T> pad_map (NodeptrT<T> node,
	const std::pair<teq::DimT,teq::DimT>& padding,
	teq::RankT dimension)
{
	if (dimension >= teq::rank_cap)
	{
		logs::fatalf("cannot pad dimension %d beyond rank_cap %d",
			dimension, teq::rank_cap);
	}
	teq::CoordT paddings;
	std::fill(paddings.begin(), paddings.end(), teq::rank_cap);
	paddings[0] = padding.first;
	paddings[1] = padding.second;
	paddings[2] = dimension;
	return FuncArg<T>(node,
		std::make_shared<teq::CoordMap>(
			[=](teq::MatrixT fwd)
			{
				for (teq::RankT i = 0; i < teq::rank_cap; ++i)
				{
					fwd[i][i] = 1;
				}
				fwd[teq::rank_cap][dimension] =
					padding.first + padding.second;
			}),
		std::make_shared<CoordMap>(paddings, false));
}

}

#endif // ETEQ_FUNCARG_HPP
