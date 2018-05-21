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
state_(data, shape, type), data_(data)
{
	if (nullptr == data ||
		false == shape.is_fully_defined() ||
		clay::DTYPE::BAD == type)
	{
		throw std::exception(); // todo: add context
	}
}

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

Constant* make_one (clay::DTYPE dtype)
{
	unsigned short bsize = clay::type_size(dtype);
	std::shared_ptr<char> ptr = clay::make_char(bsize);
	switch (dtype)
	{
		case clay::DTYPE::DOUBLE:
		{
			double d = 1;
			std::memcpy(ptr.get(), &d, bsize);
		}
		break;
		case clay::DTYPE::FLOAT:
		{
			float d = 1;
			std::memcpy(ptr.get(), &d, bsize);
		}
		break;
		case clay::DTYPE::INT8:
		{
			int8_t d = 1;
			std::memcpy(ptr.get(), &d, bsize);
		}
		break;
		case clay::DTYPE::UINT8:
		{
			uint8_t d = 1;
			std::memcpy(ptr.get(), &d, bsize);
		}
		break;
		case clay::DTYPE::INT16:
		{
			int16_t d = 1;
			std::memcpy(ptr.get(), &d, bsize);
		}
		break;
		case clay::DTYPE::UINT16:
		{
			uint16_t d = 1;
			std::memcpy(ptr.get(), &d, bsize);
		}
		break;
		case clay::DTYPE::INT32:
		{
			int32_t d = 1;
			std::memcpy(ptr.get(), &d, bsize);
		}
		break;
		case clay::DTYPE::UINT32:
		{
			uint32_t d = 1;
			std::memcpy(ptr.get(), &d, bsize);
		}
		break;
		case clay::DTYPE::INT64:
		{
			int64_t d = 1;
			std::memcpy(ptr.get(), &d, bsize);
		}
		break;
		case clay::DTYPE::UINT64:
		{
			uint64_t d = 1;
			std::memcpy(ptr.get(), &d, bsize);
		}
		default:
		break;
	}
	return new Constant(ptr, clay::Shape({1}), dtype);
}

}

#endif
