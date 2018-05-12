//
//  error.cpp
//  ioutil
//

#include <functional>
#include <numeric>

#include "ioutil/error.hpp"

#ifdef IOUTIL_ERROR_HPP

namespace ioutil
{

const char* Error::what (void) const throw ()
{
	return this->str().c_str(); 
}

}

#endif
