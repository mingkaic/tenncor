#include <cstring>

#include "soil/constant.hpp"

#ifdef CONSTANT_HPP

Nodeptr Constant::get (char* data, DTYPE type, Shape shape)
{
	return Nodeptr(new Constant(data, type, shape));
}

std::shared_ptr<char> Constant::calculate (void)
{
	if (data_ == nullptr)
	{
		handle_error("constant data is nullptr");
	}
	return data_;
}

Nodeptr Constant::gradient (Nodeptr& leaf) const
{
	if (this == leaf.get())
	{
		handle_error("eliciting gradient from constant");
	}
	return get_zero(shape_, type_);
}

Shape Constant::shape (void) const
{
	return shape_;
}

Constant::Constant (char* data, DTYPE type, Shape shape) :
	data_(make_data(data, type_size(type) * shape.n_elems())),
	shape_(shape), type_(type) {}

Nodeptr get_zero (Shape shape, DTYPE type)
{
	uint8_t nbytes = type_size(type) * shape.n_elems();
	std::string zero(nbytes, 0);
	return Constant::get(&zero[0], type, shape);
}

Nodeptr get_one (Shape shape, DTYPE type)
{
	NElemT n = shape.n_elems();
	uint8_t nbytes = type_size(type) * n;
	std::string one(nbytes, 0);
	char* ptr = &one[0];
	switch (type)
	{
		case DTYPE::DOUBLE:
		{
			double* dptr = (double*) ptr;
			std::fill(dptr, dptr + n, 1);
		}
		break;
		case DTYPE::FLOAT:
		{
			float* dptr = (float*) ptr;
			std::fill(dptr, dptr + n, 1);
		}
		break;
		case DTYPE::INT8:
		{
			int8_t* dptr = (int8_t*) ptr;
			std::fill(dptr, dptr + n, 1);
		}
		break;
		case DTYPE::UINT8:
		{
			uint8_t* dptr = (uint8_t*) ptr;
			std::fill(dptr, dptr + n, 1);
		}
		break;
		case DTYPE::INT16:
		{
			int16_t* dptr = (int16_t*) ptr;
			std::fill(dptr, dptr + n, 1);
		}
		break;
		case DTYPE::UINT16:
		{
			uint16_t* dptr = (uint16_t*) ptr;
			std::fill(dptr, dptr + n, 1);
		}
		break;
		case DTYPE::INT32:
		{
			int32_t* dptr = (int32_t*) ptr;
			std::fill(dptr, dptr + n, 1);
		}
		break;
		case DTYPE::UINT32:
		{
			uint32_t* dptr = (uint32_t*) ptr;
			std::fill(dptr, dptr + n, 1);
		}
		break;
		case DTYPE::INT64:
		{
			int64_t* dptr = (int64_t*) ptr;
			std::fill(dptr, dptr + n, 1);
		}
		break;
		case DTYPE::UINT64:
		{
			uint64_t* dptr = (uint64_t*) ptr;
			std::fill(dptr, dptr + n, 1);
		}
		default:
		break;
	}
	return Constant::get(ptr, type, shape);
}

Nodeptr get_identity (DimT d, DTYPE type, Shape inner)
{
	Shape outs({inner, Shape({d, d})});
	uint8_t bsize = type_size(type);
	NElemT n = inner.n_elems();
	uint64_t nbytes = bsize * n;
	std::string one(nbytes, 0);
	char* ptr = &one[0];
	switch (type)
	{
		case DTYPE::DOUBLE:
		{
			double* dptr = (double*) ptr;
			std::fill(dptr, dptr + n, 1);
		}
		break;
		case DTYPE::FLOAT:
		{
			float* dptr = (float*) ptr;
			std::fill(dptr, dptr + n, 1);
		}
		break;
		case DTYPE::INT8:
		{
			int8_t* dptr = (int8_t*) ptr;
			std::fill(dptr, dptr + n, 1);
		}
		break;
		case DTYPE::UINT8:
		{
			uint8_t* dptr = (uint8_t*) ptr;
			std::fill(dptr, dptr + n, 1);
		}
		break;
		case DTYPE::INT16:
		{
			int16_t* dptr = (int16_t*) ptr;
			std::fill(dptr, dptr + n, 1);
		}
		break;
		case DTYPE::UINT16:
		{
			uint16_t* dptr = (uint16_t*) ptr;
			std::fill(dptr, dptr + n, 1);
		}
		break;
		case DTYPE::INT32:
		{
			int32_t* dptr = (int32_t*) ptr;
			std::fill(dptr, dptr + n, 1);
		}
		break;
		case DTYPE::UINT32:
		{
			uint32_t* dptr = (uint32_t*) ptr;
			std::fill(dptr, dptr + n, 1);
		}
		break;
		case DTYPE::INT64:
		{
			int64_t* dptr = (int64_t*) ptr;
			std::fill(dptr, dptr + n, 1);
		}
		break;
		case DTYPE::UINT64:
		{
			uint64_t* dptr = (uint64_t*) ptr;
			std::fill(dptr, dptr + n, 1);
		}
		default:
		break;
	}
	std::string identity(nbytes * d * d, 0);
	char* iptr = &identity[0];
	for (DimT i = 0; i < d; ++i)
	{
		std::memcpy(iptr + i * (nbytes * (d + 1)), ptr, nbytes);
	}
	return Constant::get(iptr, type, outs);
}

#endif
