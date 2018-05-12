//
//  const_init.cpp
//  kiln
//

#include "kiln/const_init.hpp"

#ifdef KILN_CONST_INIT_HPP

namespace kiln
{

ConstInit::ConstInit (Validator validate) :
	Builder(validate) {}

ConstInit::ConstInit (std::string data, clay::DTYPE dtype, Validator validate) :
	Builder(validate, dtype), data_(data) {}

void ConstInit::init (char* dest, size_t nbytes) const
{
	size_t ncopied = data_.size();
	memcpy(dest, data_.c_str(), std::min(nbytes, ncopied));
	for (; ncopied * 2 <= nbytes; ncopied *= 2)
	{
		memcpy(dest + ncopied, dest, ncopied);
	}
	if(ncopied < nbytes)
	{
		memcpy(dest + ncopied, dest, nbytes - ncopied);
	}
}

}

#endif
