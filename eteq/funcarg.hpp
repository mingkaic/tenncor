#include "teq/funcarg.hpp"

#include "eteq/coord.hpp"
#include "eteq/inode.hpp"

#ifndef ETEQ_FUNCARG_HPP
#define ETEQ_FUNCARG_HPP

namespace eteq
{

template <typename T>
struct FuncArg final
{
	/// Construct FuncArg with specific coorder_ and map_io_ flag
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

template <typename T>
using ArgsT = std::vector<FuncArg<T>>;

template <typename T>
FuncArg<T> identity_map (NodeptrT<T> node)
{
	return FuncArg<T>(node, teq::identity, nullptr);
}

template <typename T>
FuncArg<T> reduce_map (NodeptrT<T> node, teq::RankT offset, teq::RankT ndims)
{
	if (offset >= teq::rank_cap)
	{
		logs::fatalf("cannot dimensions [%d,...] greater or equal to %d",
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

template <typename T>
FuncArg<T> extend_map (NodeptrT<T> node,
	teq::RankT rank, std::vector<teq::DimT> ext)
{
	return FuncArg<T>(node, teq::extend(rank, ext), extend(rank, ext));
}

template <typename T>
FuncArg<T> permute_map (NodeptrT<T> node, std::vector<teq::RankT> order)
{
	return FuncArg<T>(node, teq::permute(order), permute(order));
}

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
