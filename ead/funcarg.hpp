#include "ade/funcarg.hpp"

#include "ead/coord.hpp"
#include "ead/inode.hpp"

#ifndef EAD_FUNCARG_HPP
#define EAD_FUNCARG_HPP

namespace ead
{

template <typename T>
struct FuncArg final
{
	/// Construct FuncArg with specific coorder_ and map_io_ flag
	FuncArg (NodeptrT<T> node, ade::CoordptrT shaper, CoordptrT coorder) :
		node_(node), shaper_(shaper), coorder_(coorder)
	{
		if (node_ == nullptr)
		{
			logs::fatal("cannot map a null node");
		}
	}

	/// Return shape of tensor filtered through coordinate mapper
	ade::Shape shape (void) const
	{
		return ade::apply_shaper(shaper_, node_->get_tensor()->shape());
	}

	/// Return tensor being mapped
	ade::TensptrT get_tensor (void) const
	{
		return node_->get_tensor();
	}

	NodeptrT<T> get_node (void) const
	{
		return node_;
	}

	/// Return shaper coord map
	ade::CoordptrT get_shaper (void) const
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
	ade::CoordptrT shaper_;

	/// Coordinate mapper
	CoordptrT coorder_;
};

template <typename T>
using ArgsT = std::vector<FuncArg<T>>;

template <typename T>
FuncArg<T> identity_map (NodeptrT<T> node)
{
	return FuncArg<T>(node, ade::identity, nullptr);
}

template <typename T>
FuncArg<T> reduce_map (NodeptrT<T> node, ade::RankT offset, ade::RankT ndims)
{
	if (offset >= ade::rank_cap)
	{
		logs::fatalf("cannot dimensions [%d,...] greater or equal to %d",
			offset, ade::rank_cap);
	}

	ade::RankT n = std::min<ade::RankT>(offset + ndims, ade::rank_cap);
	ade::Shape shape = node->get_tensor()->shape();
	std::vector<ade::RankT> dims; // dims are allowed to be non-contiguous
	std::vector<ade::DimT> slist;
	dims.reserve(n);
	slist.reserve(n);

	for (ade::RankT i = offset; i < n; ++i)
	{
		if (shape.at(i) > 1)
		{
			dims.push_back(i);
		}
		slist.push_back(shape.at(i));
	}

	return FuncArg<T>(node, ade::reduce(offset, slist), reduce(dims));
}

template <typename T>
FuncArg<T> extend_map (NodeptrT<T> node,
	ade::RankT rank, std::vector<ade::DimT> ext)
{
	return FuncArg<T>(node, ade::extend(rank, ext), extend(rank, ext));
}

template <typename T>
FuncArg<T> permute_map (NodeptrT<T> node, std::vector<ade::RankT> order)
{
	return FuncArg<T>(node, ade::permute(order), permute(order));
}

template <typename T>
FuncArg<T> slice_map (NodeptrT<T> node, ade::RankT offset, 
	ade::RankT extent, ade::RankT dimension)
{
	if (dimension >= ade::rank_cap)
	{
		logs::fatalf("cannot slice dimension %d beyond rank_cap %d",
			dimension, ade::rank_cap);
	}
	ade::CoordT slicings;
	std::fill(slicings.begin(), slicings.end(), ade::rank_cap);
	slicings[0] = offset;
	slicings[1] = extent;
	slicings[2] = dimension;
	return FuncArg<T>(node,
		std::make_shared<ade::CoordMap>(
			[=](ade::MatrixT fwd)
			{
				for (ade::RankT i = 0; i < ade::rank_cap; ++i)
				{
					fwd[i][i] = 1;
				}
				fwd[ade::rank_cap][dimension] =
					extent - node->shape().at(dimension);
			}),
		std::make_shared<CoordMap>(slicings, false));
}

template <typename T>
FuncArg<T> pad_map (NodeptrT<T> node, 
	const std::pair<ade::DimT,ade::DimT>& padding, 
	ade::RankT dimension)
{
	if (dimension >= ade::rank_cap)
	{
		logs::fatalf("cannot pad dimension %d beyond rank_cap %d",
			dimension, ade::rank_cap);
	}
	ade::CoordT paddings;
	std::fill(paddings.begin(), paddings.end(), ade::rank_cap);
	paddings[0] = padding.first;
	paddings[1] = padding.second;
	paddings[2] = dimension;
	return FuncArg<T>(node,
		std::make_shared<ade::CoordMap>(
			[=](ade::MatrixT fwd)
			{
				for (ade::RankT i = 0; i < ade::rank_cap; ++i)
				{
					fwd[i][i] = 1;
				}
				fwd[ade::rank_cap][dimension] =
					padding.first + padding.second;
			}),
		std::make_shared<CoordMap>(paddings, false));
}

}

#endif // EAD_FUNCARG_HPP
