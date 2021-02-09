#include "internal/eigen/convert.hpp"

#ifdef EIGEN_CONVERT_HPP

namespace eigen
{

DimensionsT shape_convert (teq::Shape shape)
{
	DimensionsT dims;
	auto slist = shape.to_list();
	std::copy(slist.begin(), slist.end(), dims.begin());
	return dims;
}

}

#endif
