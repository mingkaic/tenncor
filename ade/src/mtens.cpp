#include "ade/mtens.hpp"

#ifdef ADE_MTENS_HPP

namespace ade
{

MappedTensor identity_map (TensptrT tensor)
{
	return MappedTensor(tensor, identity);
}

MappedTensor reduce_1d_map (TensptrT tensor, uint8_t rank)
{
	Shape shape = tensor->shape();
	std::vector<DimT> indices(rank_cap);
	auto bt = indices.begin();
	auto it = bt + rank;
	std::iota(bt, it, 0);
	std::iota(it, indices.end(), rank + 1);
	indices[rank_cap - 1] = rank;
	return MappedTensor(tensor, CoordptrT(
		reduce(rank, {shape.at(rank)})->
		connect(*permute(indices))));
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
