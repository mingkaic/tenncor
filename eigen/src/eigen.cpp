#include "eigen/eigen.hpp"

#ifdef EIGEN_EIGEN_HPP

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
