#include "ead/tensor.hpp"

#ifdef EAD_TENSOR_HPP

namespace ead
{

DimensionsT shape_convert (ade::Shape shape)
{
	DimensionsT dims;
	std::copy(shape.begin(), shape.end(), dims.begin());
	return dims;
}

}

#endif
