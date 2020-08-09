#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include "teq/teq.hpp"

#ifndef PYTUTIL_CONVERT_HPP
#define PYTUTIL_CONVERT_HPP

namespace pyutils
{

namespace py = pybind11;

std::vector<teq::DimT> c2pshape (const teq::Shape& cshape);

teq::Shape p2cshape (const py::list& pyshape);

teq::Shape p2cshape (const py::ssize_t* pslist, size_t ndim);

template <typename T>
py::array shapedarr2arr (const teq::ShapedArr<T>& sarr)
{
	auto pshape = c2pshape(sarr.shape_);
	return py::array(
		py::array::ShapeContainer(pshape.begin(), pshape.end()),
		sarr.data_.data());
}

template <typename T>
void arr2shapedarr (teq::ShapedArr<T>& out, py::array& data)
{
	auto dtype = data.dtype();
	const void* dptr = data.data();
	const py::ssize_t* sptr = data.shape();
	size_t ndim = data.ndim();
	out.shape_ = p2cshape(sptr, ndim);
	size_t n = out.shape_.n_elems();
	char kind = dtype.kind();
	py::ssize_t tbytes = dtype.itemsize();
	switch (kind)
	{
		case 'f':
			switch (tbytes)
			{
				case 4: // float32
				{
					const float* fptr = static_cast<const float*>(dptr);
					out.data_ = std::vector<T>(fptr, fptr + n);
				}
				break;
				case 8: // float64
				{
					const double* fptr = static_cast<const double*>(dptr);
					out.data_ = std::vector<T>(fptr, fptr + n);
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
					out.data_ = std::vector<T>(iptr, iptr + n);
				}
					break;
				case 2: // int16
				{
					const int16_t* iptr = static_cast<const int16_t*>(dptr);
					out.data_ = std::vector<T>(iptr, iptr + n);
				}
					break;
				case 4: // int32
				{
					const int32_t* iptr = static_cast<const int32_t*>(dptr);
					out.data_ = std::vector<T>(iptr, iptr + n);
				}
					break;
				case 8: // int64
				{
					const int64_t* iptr = static_cast<const int64_t*>(dptr);
					out.data_ = std::vector<T>(iptr, iptr + n);
				}
					break;
				default:
					global::fatalf("unsupported integer type with %d bytes", tbytes);
			}
			break;
		default:
			global::fatalf("unknown dtype %c", kind);
	}
}

}

#endif // PYTUTIL_CONVERT_HPP
