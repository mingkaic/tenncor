#include "ade/itensor.hpp"
#include "ade/coord.hpp"

#ifndef ADE_CMAP_HPP
#define ADE_CMAP_HPP

namespace ade
{

/// Coordinate mapper and tensor pair
struct MappedTensor final
{
	/// Construct MappedTensor auto deducing coorder_ and map_io_ flag
	MappedTensor (TensptrT tensor, CoordptrT shaper) :
		tensor_(tensor), shaper_(shaper)
	{
		if (tensor_ == nullptr)
		{
			logs::fatal("cannot map a null tensor");
		}
		map_io_ = tensor_->shape().n_elems() > shape().n_elems();
		if (shaper == identity || map_io_)
		{
			coorder_ = shaper;
		}
		else
		{
			coorder_ = CoordptrT(shaper->reverse());
		}
	}

	/// Construct MappedTensor with specific coorder_ and map_io_ flag
	MappedTensor (TensptrT tensor, CoordptrT shaper,
		bool map_io, CoordptrT coorder) :
		tensor_(tensor), shaper_(shaper),
		map_io_(map_io), coorder_(coorder)
	{
		if (tensor_ == nullptr)
		{
			logs::fatal("cannot map a null tensor");
		}
	}

	/// Return shape of tensor filtered through coordinate mapper
	Shape shape (void) const
	{
		ade::Shape shape = tensor_->shape();
		CoordT out;
		CoordT in;
		std::copy(shape.begin(), shape.end(), in.begin());
		shaper_->forward(out.begin(), in.begin());
		std::vector<DimT> slist(rank_cap);
		std::transform(out.begin(), out.end(), slist.begin(),
			[](CDimT cd) -> DimT
			{
				if (cd < 0)
				{
					cd = -cd - 1;
				}
				return std::round(cd);
			});
		return Shape(slist);
	}

	TensptrT get_tensor (void) const
	{
		return tensor_;
	}

	CoordptrT get_shaper (void) const
	{
		return shaper_;
	}

	/// Return map_io_ flag
	bool map_io (void) const
	{
		return map_io_;
	}

	CoordptrT get_coorder (void) const
	{
		return coorder_;
	}

private:
	/// Tensor reference
	TensptrT tensor_;

	/// Shape mapper
	CoordptrT shaper_;

	/// True if map input coordinate to output, False otherwise
	/// (if n_elems of inputshape > n_elems of outputshape)
	bool map_io_;

	/// Coordinate mapper
	CoordptrT coorder_;
};

MappedTensor identity_map (TensptrT tensor);

MappedTensor reduce_map (TensptrT tensor,
	uint8_t rank, std::vector<DimT> red);

MappedTensor extend_map (TensptrT tensor,
	uint8_t rank, std::vector<DimT> ext);

MappedTensor permute_map (TensptrT tensor, std::vector<uint8_t> order);

MappedTensor flip_map (TensptrT tensor, uint8_t dim);

}

#endif // ADE_CMAP_HPP
