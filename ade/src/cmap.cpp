#include "ade/cmap.hpp"

#ifdef ADE_CMAP_HPP

namespace ade
{

static Shape calc_shape (CoordptrT shaper, const Shape& shape)
{
	CoordT out;
	CoordT in;
	std::copy(shape.begin(), shape.end(), in.begin());
	shaper->forward(out.begin(), in.begin());
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

Shape MappedTensor::shape (void) const
{
	return calc_shape(shaper_, tensor_->shape());
}

MappedTensor MappedTensor::connect (MappedTensor lhs) const
{
	CoordptrT outshaper(shaper_->connect(*lhs.get_shaper()));
	Shape inshape = tensor_->shape();
	Shape outshape = calc_shape(outshaper, inshape);
	bool outmap_io = inshape.n_elems() > outshape.n_elems();
	CoordptrT rhs_coorder = outmap_io == map_io_ ? coorder_ :
		CoordptrT(coorder_->reverse());
	CoordptrT lhs_coorder = lhs.get_coorder();
	if (outmap_io != lhs.map_io())
	{
		lhs_coorder = CoordptrT(lhs_coorder->reverse());
	}
	CoordptrT outcoorder(outmap_io ? rhs_coorder->connect(*lhs_coorder) :
		lhs_coorder->connect(*rhs_coorder));
	return MappedTensor(tensor_, outshaper, outmap_io, outcoorder);
}

MappedTensor identity_map (TensptrT tensor)
{
	return MappedTensor(tensor, ade::identity);
}

MappedTensor reduce_map (TensptrT tensor, uint8_t rank, std::vector<DimT> red)
{
	return MappedTensor(tensor, reduce(rank, red));
}

MappedTensor extend_map (TensptrT tensor, uint8_t rank, std::vector<DimT> ext)
{
	return MappedTensor(tensor, extend(rank, ext));
}

MappedTensor permute_map (TensptrT tensor, std::vector<uint8_t> order)
{
	return MappedTensor(tensor, permute(order));
}

MappedTensor flip_map (TensptrT tensor, uint8_t dim)
{
	return MappedTensor(tensor, flip(dim));
}

}

#endif
