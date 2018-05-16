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

iNode* make_one (clay::DTYPE dtype)
{
	iNode* out = nullptr;
	switch (dtype)
	{
		case clay::DTYPE::DOUBLE:
			out = make_constant((double) 1);
		break;
		case clay::DTYPE::FLOAT:
			out = make_constant((float) 1);
		break;
		case clay::DTYPE::INT8:
			out = make_constant((int8_t) 1);
		break;
		case clay::DTYPE::INT16:
			out = make_constant((int16_t) 1);
		break;
		case clay::DTYPE::INT32:
			out = make_constant((int32_t) 1);
		break;
		case clay::DTYPE::INT64:
			out = make_constant((int64_t) 1);
		break;
		case clay::DTYPE::UINT8:
			out = make_constant((uint8_t) 1);
		break;
		case clay::DTYPE::UINT16:
			out = make_constant((uint16_t) 1);
		break;
		case clay::DTYPE::UINT32:
			out = make_constant((uint32_t) 1);
		break;
		case clay::DTYPE::UINT64:
			out = make_constant((uint64_t) 1);
		break;
		default:
		break;
	}
	return out;
}

}

#endif
