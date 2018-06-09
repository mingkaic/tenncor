//
//  norm_init.cpp
//  kiln
//

#include "kiln/norm_init.hpp"

#include "clay/memory.hpp"

#include "slip/registry.hpp"

#ifdef KILN_NORM_INIT_HPP

namespace kiln
{

clay::Tensor* norm_build (char* mean, char* stdev,
	clay::Shape shape, clay::DTYPE dtype)
{
	unsigned short bsize = clay::type_size(dtype);
	size_t nbytes = bsize * shape.n_elems();
	std::shared_ptr<char> mean_ptr = clay::make_char(nbytes);
	std::shared_ptr<char> stdev_ptr = clay::make_char(bsize);

	char* dest = mean_ptr.get();
	copy_over(dest, nbytes, mean, bsize);
	std::memcpy(stdev_ptr.get(), stdev, bsize);

	clay::Shape one({1});
	mold::iOperatePtrT op = slip::get_op(slip::NORM);
	std::vector<clay::State> states{
		clay::State{mean_ptr, shape, dtype},
		clay::State{stdev_ptr, one, dtype},
	};
	mold::ImmPair imm = op->get_imms(states);
	auto out = new clay::Tensor(imm.first, imm.second);
	clay::State state = out->get_state();
	op->write_data(state, states);
	return out;
}

}

#endif
