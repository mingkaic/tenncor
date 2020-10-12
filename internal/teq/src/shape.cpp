#include "internal/teq/shape.hpp"

#ifdef TEQ_SHAPE_HPP

namespace teq
{

std::vector<DimT> narrow_shape (const Shape& sign)
{
	auto it = sign.begin(), et = sign.end();
	while (it != et && *(et - 1) <= 1)
	{
		--et;
	}
	return std::vector<DimT>(it, et);
}

}

#endif
