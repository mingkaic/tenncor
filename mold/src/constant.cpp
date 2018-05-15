//
//  constant.cpp
//  mold
//

#include "mold/constant.hpp"

#ifdef MOLD_CONSTANT_HPP

namespace mold
{

Constant::Constant (std::shared_ptr<char> data,
	clay::Shape shape, clay::DTYPE type) :
state_(data, shape, type), data_(data) {}

bool Constant::has_data (void) const
{
	return true;
}

clay::State Constant::get_state (void) const
{
	return state_;
}

iNode* Constant::derive (iNode* wrt)
{
	clay::DTYPE dtype = state_.dtype_;
	unsigned short bsize = clay::type_size(dtype);
	std::shared_ptr<char> out = clay::make_char(bsize);
	memset(out.get(), 0, bsize);
	return new Constant(out, clay::Shape(std::vector<size_t>{1}), dtype);
}

}

#endif
