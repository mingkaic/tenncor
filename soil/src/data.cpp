#include <cstring>

#include "soil/error.hpp"
#include "soil/data.hpp"

#ifdef DATA_HPP

struct varr_deleter
{
	void operator () (void* p)
	{
		free(p);
	}
};

std::shared_ptr<char> make_data (size_t nbytes)
{
	return std::shared_ptr<char>(
		(char*) malloc(nbytes), varr_deleter());
}

std::shared_ptr<char> make_data (char* data, size_t nbytes)
{
	std::shared_ptr<char> out = std::shared_ptr<char>(
		(char*) malloc(nbytes), varr_deleter());
	std::memcpy(out.get(), data, nbytes);
	return out;
}

#endif
