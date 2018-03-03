#include "include/utils/shared_ptr.hpp"

#ifdef TENNCOR_SHARED_PTR_HPP

namespace nnutils
{

struct varr_deleter
{
	void operator () (void* p)
	{
		free(p);
	}
};

std::shared_ptr<void> make_svoid (size_t nbytes)
{
	return std::shared_ptr<void>(malloc(nbytes), varr_deleter());
}

void check_ptr (std::shared_ptr<void>& ptr, size_t nbytes)
{
	if (nullptr == ptr)
	{
		ptr = make_svoid(nbytes);
	}
}

}

#endif
