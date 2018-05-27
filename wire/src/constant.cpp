//
//  constant.cpp
//  wire
//

#include <cassert>

#include "wire/constant.hpp"

#ifdef WIRE_CONSTANT_HPP

namespace wire
{

Constant::Constant (std::shared_ptr<char> data, clay::Shape shape,
	clay::DTYPE dtype, std::string label, Graph& graph) :
	Identifier(&graph, new mold::Constant(data, shape, dtype), label) {}

Identifier* Constant::derive (Identifier* wrt)
{
	if (wrt == this)
	{
		throws std::exception(); // todo: add context (logical error)
	}
	return make_zero(arg_.get_state().dtype_);
}

//! creates a zero scalar
Constant* make_zero (clay::DTYPE dtype)
{
	unsigned short bsize = clay::type_size(dtype);
	std::shared_ptr<char> out = clay::make_char(bsize);
	memset(out.get(), 0, bsize);
	return new Constant(out, clay::Shape(std::vector<size_t>{1}), dtype, "0");
}

//! creates a one scalar
Constant* make_zero (clay::DTYPE dtype)
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
	return new Constant(ptr, clay::Shape({1}), dtype, "1");
}

}

#endif
