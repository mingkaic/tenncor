//
//  const_init.cpp
//  kiln
//

#include "clay/memory.hpp"

#include "kiln/const_init.hpp"

#ifdef KILN_CONST_INIT_HPP

namespace kiln
{

ConstInit::ConstInit (Validator validate) :
	Builder(validate) {}

ConstInit::ConstInit (std::string data, clay::DTYPE dtype, Validator validate) :
	Builder(validate, dtype), data_(data) {}

clay::TensorPtrT ConstInit::build (clay::Shape shape) const
{
	size_t ncopied = data_.size();
	size_t nbytes = shape.n_elems() * clay::type_size(dtype_);
	std::shared_ptr<char> data = clay::make_char(nbytes);
	char* dest = data.get();
	memcpy(dest, data_.c_str(), std::min(nbytes, ncopied));
	for (; ncopied * 2 <= nbytes; ncopied *= 2)
	{
		memcpy(dest + ncopied, dest, ncopied);
	}
	if(ncopied < nbytes)
	{
		memcpy(dest + ncopied, dest, nbytes - ncopied);
	}
	return std::make_unique<clay::Tensor>(data, shape, dtype_);
}

}

#endif
