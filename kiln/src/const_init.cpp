//
//  const_init.cpp
//  kiln
//

#include "kiln/const_init.hpp"

#ifdef KILN_CONST_INIT_HPP

namespace kiln
{

void copy_over (char* dest, size_t ndest,
	const char* src, size_t nsrc)
{
	memcpy(dest, src, std::min(ndest, nsrc));
	for (; nsrc * 2 <= ndest; nsrc *= 2)
	{
		memcpy(dest + nsrc, dest, nsrc);
	}
	if (nsrc < ndest)
	{
		memcpy(dest + nsrc, dest, ndest - nsrc);
	}
}

void correct_shape(clay::Shape& shape, size_t n)
{
	if (false == shape.is_fully_defined())
	{
		if (shape.rank() == 0)
		{
			shape = {1};
		}
		else
		{
			std::vector<size_t> slist = shape.as_list();
			for (size_t& s : slist)
			{
				if (0 == s)
				{
					s = 1;
				}
			}
			shape = slist;
		}
	}
}

}

#endif
