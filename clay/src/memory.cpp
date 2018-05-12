//
//  memory.cpp
//  clay
//

#include "clay/memory.hpp"

#ifdef CLAY_MEMORY_HPP

namespace clay
{

struct varr_deleter
{
	void operator () (void* p)
	{
		free(p);
	}
};

std::shared_ptr<char> make_char (size_t n)
{
	return std::shared_ptr<char>((char*) malloc(n), varr_deleter());
}

}

#endif 
