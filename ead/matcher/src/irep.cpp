#include "ead/matcher/irep.hpp"

#ifdef OPT_IREP_HPP

namespace opt
{

std::string encode_tens (ade::iTensor* tens)
{
	// pointer to string
	size_t iptr = (size_t) tens;
	return fmts::to_string(iptr);
}

std::string encode_coorder (ade::iCoordMap* coorder)
{
	// pointer to string
	size_t iptr = (size_t) coorder;
	return fmts::to_string(iptr);
}

}

#endif
