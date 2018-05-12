//
//  validator.cpp
//  kiln
//

#include "kiln/validator.hpp"

#ifdef KILN_VALIDATOR_HPP

namespace kiln
{

Validator::Validator (clay::Shape allowed, std::unordered_set<clay::DTYPE> reject) :
	allowed_(allowed), reject_(reject) {}

bool Validator::support (clay::Shape shape, clay::DTYPE dtype) const
{
	return shape.is_fully_defined() &&
	shape.is_compatible_with(allowed_) &&
	dtype != clay::DTYPE::BAD &&
	reject_.end() == reject_.find(dtype);
}

}

#endif
