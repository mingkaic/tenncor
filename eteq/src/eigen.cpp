#include "eteq/eigen.hpp"

#ifdef ETEQ_EIGEN_HPP

namespace eteq
{

DimensionsT shape_convert (teq::Shape shape)
{
	DimensionsT dims;
	std::copy(shape.begin(), shape.end(), dims.begin());
	return dims;
}

}

#endif
