//
//  unif_init.cpp
//  kiln
//

#include "kiln/unif_init.hpp"

#include "clay/memory.hpp"

#include "slip/registry.hpp"

#ifdef KILN_UNIF_INIT_HPP

namespace kiln
{

clay::TensorPtrT unif_build (char* min, char* max,
	clay::Shape shape, clay::DTYPE dtype)
{
	unsigned short bsize = clay::type_size(dtype);
	size_t nbytes = bsize * shape.n_elems();
	std::shared_ptr<char> min_ptr = clay::make_char(nbytes);
	std::shared_ptr<char> max_ptr = clay::make_char(bsize);

	char* dest = min_ptr.get();
	copy_over(dest, nbytes, min, bsize);
	std::memcpy(max_ptr.get(), max, bsize);

	clay::Shape one({1});
	return slip::get_op(slip::UNIF)->make_data({
		clay::State{min_ptr, shape, dtype},
		clay::State{max_ptr, one, dtype},
	});
}

}

#endif
