#include "internal/eigen/convert.hpp"

#ifdef EIGEN_CONVERT_HPP

namespace eigen
{

DimensionsT shape_convert (teq::Shape shape)
{
	DimensionsT dims;
	std::copy(shape.begin(), shape.end(), dims.begin());
	return dims;
}

}

#endif
