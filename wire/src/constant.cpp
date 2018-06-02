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
		throw std::logic_error("deriving with respect to a constant");
	}
	clay::State state = args_[0]->get_state();
	return make_zero(state.shape_, state.dtype_);
}

//! creates a zero scalar
Constant* make_zero (clay::Shape shape, clay::DTYPE dtype)
{
	size_t nbytes = clay::type_size(dtype) * shape.n_elems();
	std::shared_ptr<char> out = clay::make_char(nbytes);
	memset(out.get(), 0, nbytes);
	return new Constant(out, shape, dtype, "0");
}

//! creates a one scalar
Constant* make_one (clay::Shape shape, clay::DTYPE dtype)
{
	size_t n = shape.n_elems();
	size_t nbytes = clay::type_size(dtype) * n;
	std::shared_ptr<char> ptr = clay::make_char(nbytes);
	switch (dtype)
	{
		case clay::DTYPE::DOUBLE:
		{
			double* dptr = (double*) ptr.get();
			std::fill(dptr, dptr + n, 1);
		}
		break;
		case clay::DTYPE::FLOAT:
		{
			float* dptr = (float*) ptr.get();
			std::fill(dptr, dptr + n, 1);
		}
		break;
		case clay::DTYPE::INT8:
		{
			int8_t* dptr = (int8_t*) ptr.get();
			std::fill(dptr, dptr + n, 1);
		}
		break;
		case clay::DTYPE::UINT8:
		{
			uint8_t* dptr = (uint8_t*) ptr.get();
			std::fill(dptr, dptr + n, 1);
		}
		break;
		case clay::DTYPE::INT16:
		{
			int16_t* dptr = (int16_t*) ptr.get();
			std::fill(dptr, dptr + n, 1);
		}
		break;
		case clay::DTYPE::UINT16:
		{
			uint16_t* dptr = (uint16_t*) ptr.get();
			std::fill(dptr, dptr + n, 1);
		}
		break;
		case clay::DTYPE::INT32:
		{
			int32_t* dptr = (int32_t*) ptr.get();
			std::fill(dptr, dptr + n, 1);
		}
		break;
		case clay::DTYPE::UINT32:
		{
			uint32_t* dptr = (uint32_t*) ptr.get();
			std::fill(dptr, dptr + n, 1);
		}
		break;
		case clay::DTYPE::INT64:
		{
			int64_t* dptr = (int64_t*) ptr.get();
			std::fill(dptr, dptr + n, 1);
		}
		break;
		case clay::DTYPE::UINT64:
		{
			uint64_t* dptr = (uint64_t*) ptr.get();
			std::fill(dptr, dptr + n, 1);
		}
		default:
		break;
	}
	return new Constant(ptr, shape, dtype, "1");
}

}

#endif
