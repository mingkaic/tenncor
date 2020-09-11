
#ifndef PYTUTIL_CONVERT_HPP
#define PYTUTIL_CONVERT_HPP

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include "internal/teq/teq.hpp"

namespace pyutils
{

namespace py = pybind11;

teq::DimsT c2pshape (const teq::Shape& cshape);

teq::Shape p2cshape (const py::list& pyshape);

teq::Shape p2cshape (const py::ssize_t* pslist, size_t ndim);

template <typename T>
std::vector<T> arr2shapedarr (teq::Shape& shape, py::array& data)
{
	auto dtype = data.dtype();
	const void* dptr = data.data();
	const py::ssize_t* sptr = data.shape();
	size_t ndim = data.ndim();
	shape = p2cshape(sptr, ndim);
	size_t n = shape.n_elems();
	char kind = dtype.kind();
	py::ssize_t tbytes = dtype.itemsize();
	std::vector<T> out;
	switch (kind)
	{
		case 'f':
			switch (tbytes)
			{
				case 4: // float32
				{
					const float* fptr = static_cast<const float*>(dptr);
					out = std::vector<T>(fptr, fptr + n);
				}
				break;
				case 8: // float64
				{
					const double* fptr = static_cast<const double*>(dptr);
					out = std::vector<T>(fptr, fptr + n);
				}
					break;
				default:
					global::fatalf("unsupported float type with %d bytes", tbytes);
			}
			break;
		case 'i':
			switch (tbytes)
			{
				case 1: // int8
				{
					const int8_t* iptr = static_cast<const int8_t*>(dptr);
					out = std::vector<T>(iptr, iptr + n);
				}
					break;
				case 2: // int16
				{
					const int16_t* iptr = static_cast<const int16_t*>(dptr);
					out = std::vector<T>(iptr, iptr + n);
				}
					break;
				case 4: // int32
				{
					const int32_t* iptr = static_cast<const int32_t*>(dptr);
					out = std::vector<T>(iptr, iptr + n);
				}
					break;
				case 8: // int64
				{
					const int64_t* iptr = static_cast<const int64_t*>(dptr);
					out = std::vector<T>(iptr, iptr + n);
				}
					break;
				default:
					global::fatalf("unsupported integer type with %d bytes", tbytes);
			}
			break;
		default:
			global::fatalf("unknown dtype %c", kind);
	}
	return out;
}

}

#endif // PYTUTIL_CONVERT_HPP
