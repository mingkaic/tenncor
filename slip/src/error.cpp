//
//  error.cpp
//  slip
//

#include "slip/error.hpp"

#include "ioutil/stream.hpp"

#ifdef SLIP_ERROR_HPP

namespace slip
{

NoArgumentsError::NoArgumentsError (void) : std::runtime_error(
	"operating without arguments") {}

BadNArgsError::BadNArgsError (size_t nexpect, size_t ngot) :
	std::runtime_error(ioutil::Stream() <<
	"expected " << nexpect <<
	" arguments, got " << ngot) {}

UnsupportedOpcodeError::UnsupportedOpcodeError (OPCODE opcode) :
	std::runtime_error(ioutil::Stream() <<
	"unsupported opcode " << opcode) {}

ShapeMismatchError::ShapeMismatchError (clay::Shape shape, clay::Shape other) :
	std::runtime_error(ioutil::Stream() <<
	"mismatching shapes " << shape.as_list() <<
	" and " << other.as_list()) {}

TypeMismatchError::TypeMismatchError (clay::DTYPE type, clay::DTYPE other) :
	std::runtime_error(ioutil::Stream() <<
	"mismatching types " << type << " and " << other) {}

InvalidDimensionError::InvalidDimensionError (uint64_t dim, clay::Shape shape) :
	std::runtime_error(ioutil::Stream() <<
	"dimension " << dim <<
	" cannot accommodate shape " << shape.as_list()) {}

InvalidRangeError::InvalidRangeError (mold::Range range, clay::Shape shape) :
	std::runtime_error(ioutil::Stream() <<
	"range <" << range.lower_ << "," << range.upper_ <<
	"> cannot accommodate shape " << shape.as_list()) {}

}

#endif
