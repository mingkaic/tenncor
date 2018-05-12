//
//  unif_init.cpp
//  kiln
//

#include "kiln/unif_init.hpp"

#ifdef KILN_UNIF_INIT_HPP

namespace kiln
{

UnifInit::UnifInit (Validator validate) :
	Builder(validate) {}

UnifInit::UnifInit (std::string min, std::string max,
	clay::DTYPE dtype, Validator validate) :
	Builder(validate, dtype), min_(min), max_(max) {}

void UnifInit::init (char* dest, size_t nbytes) const
{
}

}

#endif
