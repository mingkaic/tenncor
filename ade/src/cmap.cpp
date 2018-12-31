#include "ade/cmap.hpp"

#ifdef ADE_CMAP_HPP

namespace ade
{

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
