//
//  norm_init.cpp
//  kiln
//

#include "slip/registry.hpp"

#include "kiln/norm_init.hpp"

#ifdef KILN_NORM_INIT_HPP

namespace kiln
{

NormInit::NormInit (Validator validate) :
	Builder(validate) {}

NormInit::NormInit (std::string mean, std::string stdev,
	clay::DTYPE dtype, Validator validate) :
	Builder(validate, dtype), mean_(mean), stdev_(stdev) {}

clay::TensorPtrT NormInit::build (clay::Shape shape) const
{
	unsigned short bsize = clay::type_size(dtype_);
	size_t nbytes = bsize * shape.n_elems();
	std::shared_ptr<char> mean_ptr = clay::make_char(nbytes);
	std::shared_ptr<char> stdev_ptr = clay::make_char(bsize);

	size_t ncopied = bsize;
	char* dest = mean_ptr.get();
	std::memcpy(dest, mean_.c_str(), bsize);
	for (; ncopied * 2 <= nbytes; ncopied *= 2)
	{
		memcpy(dest + ncopied, dest, ncopied);
	}
	if(ncopied < nbytes)
	{
		memcpy(dest + ncopied, dest, nbytes - ncopied);
	}
	std::memcpy(stdev_ptr.get(), stdev_.c_str(), bsize);

	clay::Shape one({1});
	mold::OperateIO op = slip::forward_op(slip::NORM);
	op.args_ = {
		clay::State{mean_ptr, shape, dtype_},
		clay::State{stdev_ptr, one, dtype_},
	};
	return op.get();
}

}

#endif
