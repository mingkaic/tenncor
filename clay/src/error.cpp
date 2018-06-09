//
//  error.cpp
//  clay
//

#include "clay/error.hpp"

#include "ioutil/stream.hpp"

#ifdef CLAY_ERROR_HPP

namespace clay
{

UnsupportedTypeError::UnsupportedTypeError (DTYPE type) :
	std::runtime_error(ioutil::Stream() <<
	"unsupported type " << type) {}

InvalidShapeError::InvalidShapeError (Shape shape) :
	std::runtime_error(ioutil::Stream() <<
	"unsupported shape " << shape.as_list()) {}

InvalidShapeError::InvalidShapeError (Shape shape, Shape shape2) :
	std::runtime_error(ioutil::Stream() <<
	"mismatch shapes: " << shape.as_list() <<
	" and " << shape2.as_list()) {}

}

#endif
