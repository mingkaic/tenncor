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
	FuncArg (NodeptrT<T> node, ade::CoordptrT shaper,
		bool map_io, ade::CoordptrT coorder) :
		node_(node), shaper_(shaper),
		map_io_(map_io), coorder_(coorder)
	{
		if (node_ == nullptr)
		{
			logs::fatal("cannot map a null node");
		}
	}

	/// Return shape of tensor filtered through coordinate mapper
	ade::Shape shape (void) const
	{
		ade::Shape shape = node_->get_tensor()->shape();
		ade::CoordT out;
		ade::CoordT in;
		std::copy(shape.begin(), shape.end(), in.begin());
		shaper_->forward(out.begin(), in.begin());
		std::vector<ade::DimT> slist(ade::rank_cap);
		std::transform(out.begin(), out.end(), slist.begin(),
			[](ade::CDimT cd) -> ade::DimT
			{
				if (cd < 0)
				{
					cd = -cd - 1;
				}
				return std::round(cd);
			});
		return ade::Shape(slist);
	}

	/// Return tensor being mapped
	NodeptrT<T> get_tensor (void) const
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
		return map_io_;
	}

	/// Return coord map for coordinates
	ade::CoordptrT get_coorder (void) const
	{
		return coorder_;
	}

private:
	/// Tensor reference
	NodeptrT<T> node_;

	/// Shape mapper
	ade::CoordptrT shaper_;

	/// True if map input coordinate to output, False otherwise
	/// (if n_elems of inputshape > n_elems of outputshape)
	bool map_io_;

	/// Coordinate mapper
	ade::CoordptrT coorder_;
};

template <typename T>
using ArgsT = std::vector<FuncArg<T>>;

template <typename T>
FuncArg<T> identity_map (NodeptrT<T> node)
{
	return FuncArg<T>(node, ade::identity, false, nullptr);
}

template <typename T>
FuncArg<T> reduce_map (NodeptrT<T> node,
	uint8_t rank, std::vector<uint8_t> red)
{
	ade::Shape shape = node->get_tensor()->shape();
	std::vector<ade::DimT> slist(red.size());
	std::transform(red.begin(), red.end(), slist.begin(),
		[&shape](uint8_t d)
		{
			return shape.at(d);
		});
	return FuncArg<T>(node, ade::reduce(rank, slist), true,
		reduce(rank, red));
}

template <typename T>
FuncArg<T> extend_map (NodeptrT<T> node,
	uint8_t rank, std::vector<ade::DimT> ext)
{
	return FuncArg<T>(node, ade::extend(rank, ext), true,
		extend(rank, ext));
}

template <typename T>
FuncArg<T> permute_map (NodeptrT<T> node, std::vector<uint8_t> order)
{
	return FuncArg<T>(node, ade::permute(order), true,
		permute(order));
}

}

#endif // EAD_FUNCARG_HPP
