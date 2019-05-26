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
FuncArg<T> reduce_map (NodeptrT<T> node, uint8_t offset, uint8_t ndims)
{
	size_t n = std::min<ade::DimT>(offset + ndims, ade::rank_cap);
	ade::Shape shape = node->get_tensor()->shape();
	std::vector<uint8_t> dims; // dims are allowed to be non-contiguous
	std::vector<ade::DimT> slist;
	dims.reserve(n);
	slist.reserve(n);

	for (size_t i = offset; i < n; ++i)
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
	uint8_t rank, std::vector<ade::DimT> ext)
{
	return FuncArg<T>(node, ade::extend(rank, ext), extend(rank, ext));
}

template <typename T>
FuncArg<T> permute_map (NodeptrT<T> node, std::vector<ade::DimT> order)
{
	return FuncArg<T>(node, ade::permute(order), permute(order));
}

}

#endif // EAD_FUNCARG_HPP
