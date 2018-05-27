//
//  unif_init.cpp
//  kiln
//

#include "clay/memory.hpp"
#include "slip/registry.hpp"

#include "kiln/unif_init.hpp"

#ifdef KILN_UNIF_INIT_HPP

namespace kiln
{

UnifInit::UnifInit (Validator validate) :
	Builder(validate) {}

UnifInit::UnifInit (std::string min, std::string max,
	clay::DTYPE dtype, Validator validate) :
	Builder(validate, dtype), min_(min), max_(max) {}

clay::TensorPtrT UnifInit::build (clay::Shape shape) const
{
	unsigned short bsize = clay::type_size(dtype_);
	size_t nbytes = bsize * shape.n_elems();
	std::shared_ptr<char> min_ptr = clay::make_char(nbytes);
	std::shared_ptr<char> max_ptr = clay::make_char(bsize);

	size_t ncopied = bsize;
	char* dest = min_ptr.get();
	std::memcpy(dest, min_.c_str(), bsize);
	for (; ncopied * 2 <= nbytes; ncopied *= 2)
	{
		memcpy(dest + ncopied, dest, ncopied);
	}
	if(ncopied < nbytes)
	{
		memcpy(dest + ncopied, dest, nbytes - ncopied);
	}
	std::memcpy(max_ptr.get(), max_.c_str(), bsize);

	clay::Shape one({1});
	mold::iOperatePtrT op = slip::forward_op(slip::UNIF);
	op->set_args({
		clay::State{min_ptr, shape, dtype_},
		clay::State{max_ptr, one, dtype_},
	});
	mold::ImmPair imm = op->get_imms();
	clay::Shape& shape = imm.first;
	clay::DTYPE& dtype = imm.second;
	size_t nbytes = shape.n_elems() * clay::type_size(dtype);
	std::shared_ptr<char> data = clay::make_char(nbytes);
	return std::make_unique<clay::Tensor>(data, shape, dtype);
}

}

#endif
