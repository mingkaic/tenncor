//
//  norm_init.cpp
//  kiln
//

#include "kiln/norm_init.hpp"

#ifdef KILN_NORM_INIT_HPP

namespace kiln
{

NormInit::NormInit (Validator validate) :
	Builder(validate) {}

NormInit::NormInit (std::string mean, std::string stdev,
	clay::DTYPE dtype, Validator validate) :
	Builder(validate, dtype), mean_(mean), stdev_(stdev) {}

void NormInit::init (char* dest, size_t nbytes) const
{
}

}

#endif
