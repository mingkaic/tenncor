#include <cstring>
#include <cmath>

#include "sand/operator.hpp"

#include "util/error.hpp"

#ifndef SAND_UNARY_HPP
#define SAND_UNARY_HPP

namespace sand
{

template <typename T>
void typecast (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata)
{
	Shape shape = srcs[0].shape_;
	T* destdata = (T*) dest.data_;
	char* s = srcs[0].data_;
	NElemT n = shape.n_elems();
	if (mdata[0] == mdata[1])
	{
		std::memcpy(destdata, s, n * sizeof(T));
		return;
	}
	switch (mdata[1])
	{
		case DOUBLE:
		{
			const double* ptr = (const double*) s;
			std::transform(ptr, ptr + n, destdata,
			[](double d) -> T
			{
				return d;
			});
		}
		break;
		case FLOAT:
		{
			const float* ptr = (const float*) s;
			std::transform(ptr, ptr + n, destdata,
			[](float d) -> T
			{
				return d;
			});
		}
		break;
		case INT8:
		{
			const int8_t* ptr = (const int8_t*) s;
			std::transform(ptr, ptr + n, destdata,
			[](int8_t d) -> T
			{
				return d;
			});
		}
		break;
		case UINT8:
		{
			const uint8_t* ptr = (const uint8_t*) s;
			std::transform(ptr, ptr + n, destdata,
			[](uint8_t d) -> T
			{
				return d;
			});
		}
		break;
		case INT16:
		{
			const int16_t* ptr = (const int16_t*) s;
			std::transform(ptr, ptr + n, destdata,
			[](int16_t d) -> T
			{
				return d;
			});
		}
		break;
		case UINT16:
		{
			const uint16_t* ptr = (const uint16_t*) s;
			std::transform(ptr, ptr + n, destdata,
			[](uint16_t d) -> T
			{
				return d;
			});
		}
		break;
		case INT32:
		{
			const int32_t* ptr = (const int32_t*) s;
			std::transform(ptr, ptr + n, destdata,
			[](int32_t d) -> T
			{
				return d;
			});
		}
		break;
		case UINT32:
		{
			const uint32_t* ptr = (const uint32_t*) s;
			std::transform(ptr, ptr + n, destdata,
			[](uint32_t d) -> T
			{
				return d;
			});
		}
		break;
		case INT64:
		{
			const int64_t* ptr = (const int64_t*) s;
			std::transform(ptr, ptr + n, destdata,
			[](int64_t d) -> T
			{
				return d;
			});
		}
		break;
		case UINT64:
		{
			const uint64_t* ptr = (const uint64_t*) s;
			std::transform(ptr, ptr + n, destdata,
			[](uint64_t d) -> T
			{
				return d;
			});
		}
		break;
		default:
			handle_error("casting from invalid type");
	}
}

template <typename T>
void unary (NodeInfo dest, std::vector<NodeInfo>& srcs,
	std::function<T(const T&)> f)
{
	Shape srcshape = srcs[0].shape_;
	T* destdata = (T*) dest.data_;
	T* srcdata = (T*) srcs[0].data_;
	size_t n = dest.shape_.n_elems();
	bool src_mul = srcshape.n_elems() > 1;
	for (size_t i = 0; i < n; ++i)
	{
		destdata[i] = f(srcdata[src_mul * i]);
	}
}

template <typename T>
void abs (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata)
{
	unary<T>(dest, srcs, [](const T& src) { return std::abs(src); });
}

template <>
void abs<uint8_t> (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata);

template <>
void abs<uint16_t> (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata);

template <>
void abs<uint32_t> (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata);

template <>
void abs<uint64_t> (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata);

template <typename T>
void neg (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata)
{
	unary<T>(dest, srcs, [](const T& src) { return -src; });
}

template <>
void neg<uint8_t> (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata);

template <>
void neg<uint16_t> (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata);

template <>
void neg<uint32_t> (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata);

template <>
void neg<uint64_t> (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata);

template <typename T>
void logic_not (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata)
{
	unary<T>(dest, srcs, [](const T& src) { return !src; });
}

template <typename T>
void sin (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata)
{
	unary<T>(dest, srcs, [](const T& src) { return std::sin(src); });
}

template <typename T>
void cos (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata)
{
	unary<T>(dest, srcs, [](const T& src) { return std::cos(src); });
}

template <typename T>
void tan (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata)
{
	unary<T>(dest, srcs, [](const T& src) { return std::tan(src); });
}

template <typename T>
void exp (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata)
{
	unary<T>(dest, srcs, [](const T& src) { return std::exp(src); });
}

template <typename T>
void log (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata)
{
	unary<T>(dest, srcs, [](const T& src) { return std::log(src); });
}

template <typename T>
void sqrt (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata)
{
	unary<T>(dest, srcs, [](const T& src) { return std::sqrt(src); });
}

template <typename T>
void round (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata)
{
	unary<T>(dest, srcs, [](const T& src) { return std::round(src); });
}

template <typename T>
void flip (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata)
{
	Shape shape = srcs[0].shape_;
	T* destdata = (T*) dest.data_;
	T* srcdata = (T*) srcs[0].data_;
	uint8_t dim = mdata[0];
	std::vector<DimT> slist = shape.as_list();
	std::vector<DimT> coord;
	for (NElemT i = 0, n = shape.n_elems();
		i < n; ++i)
	{
		coord = coordinate(shape, i);
		coord[dim] = slist[dim] - coord[dim] - 1;
		destdata[i] = srcdata[index(shape, coord)];
	}
}

// template <typename T>
// void static_transpose (NodeInfo dest, std::vector<NodeInfo>& srcs,
// 	MetaEncoder::MetaData mdata)
// {
// 	if (1 != srcs.size())
// 	{
// 		handle_error("transpose requires 1 arguments",
// 			ErrArg<size_t>{"num_args", srcs.size()});
// 	}

// 	NodeInfo& src = srcs[0];
// 	NElemT an = src.shape_.n_elems();
// 	NElemT n = dest.shape_.n_elems();

// 	if (an != n)
// 	{
// 		handle_error("transposing src to destination of incompatible size",
// 			ErrArg<size_t>{"ndest", n},
// 			ErrArg<size_t>{"nsrc", an});
// 	}

// 	T* destdata = (T*) dest.data_;
// 	T* srcdata = (T*) src.data_;

// 	NElemT srcx = src.shape_.group(0).n_elems();
// 	NElemT srcy = src.shape_.group(1).n_elems();

// 	// apply transformation
// 	for (NElemT srci = 0; srci < n; ++srci)
// 	{
// 		NElemT row = srci / srcx;
// 		NElemT col = srci % srcx;
// 		NElemT desti = row + col * srcy;
// 		destdata[desti] = srcdata[srci];
// 	}
// }

template <typename T>
void transpose (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata)
{
	if (1 != srcs.size())
	{
		handle_error("transpose requires 1 arguments",
			ErrArg<size_t>{"num_args", srcs.size()});
	}

	NodeInfo& src = srcs[0];
	NElemT n = src.shape_.n_elems();

	T* destdata = (T*) dest.data_;
	T* srcdata = (T*) src.data_;

	// apply transformation
	std::vector<DimT> coords;
	uint8_t ng0 = mdata[1] - mdata[0];
	uint8_t ng1 = mdata[2] - mdata[1];
	uint8_t ng2 = mdata[3] - mdata[2];
	uint8_t total = mdata[3] - mdata[0];
	DimT buffer[total];
	for (NElemT srci = 0; srci < n; ++srci)
	{
		coords = coordinate(src.shape_, srci);

		std::memcpy(buffer, &coords[mdata[2]], sizeof(DimT) * ng2);
		std::memcpy(buffer + ng2, &coords[mdata[1]], sizeof(DimT) * ng1);
		std::memcpy(buffer + ng2 + ng1, &coords[mdata[0]], sizeof(DimT) * ng0);
		std::memcpy(&coords[mdata[0]], buffer, sizeof(DimT) * total);

		NElemT desti = index(dest.shape_, coords);
		destdata[desti] = srcdata[srci];
	}
}

template <typename T>
void n_elems (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata)
{
	throw std::bad_function_call();
}

template <>
void n_elems<uint32_t> (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata);

template <typename T>
void n_dims (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata)
{
	throw std::bad_function_call();
}

template <>
void n_dims<uint8_t> (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata);

template <typename T>
void arg_reduce (NodeInfo dest, NodeInfo& src, DimT dim,
	std::function<bool(const T&,const T&)> cmp)
{
	Shape& srcshape = src.shape_;
	T* d = (T*) dest.data_;
	T* s = (T*) src.data_;
	auto it = srcshape.begin();
	std::vector<DimT> front(it, it + dim);
	std::vector<DimT> back(it + dim + 1, srcshape.end());

	NElemT nouter = std::max<NElemT>(1, Shape(front).n_elems());
	NElemT ninner = srcshape.at(dim);
	NElemT nout2 = std::max<NElemT>(1, Shape(back).n_elems());

	for (NElemT i = 0; i < nouter; ++i)
	{
		for (NElemT j = 0; j < nout2; ++j)
		{
			NElemT outidx = i + j * nouter;
			NElemT inidx = i + j * ninner * nouter;
			NElemT n = i + (j + 1) * ninner * nouter;
			NElemT out = inidx;
			for (inidx += nouter; inidx < n; inidx += nouter)
			{
				if (cmp(s[out], s[inidx]))
				{
					out = inidx;
				}
			}
			d[outidx] = out;
		}
	}
}

template <typename T>
void arg_max (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata)
{
	if (mdata[0] > 0)
	{
		arg_reduce<T>(dest, srcs[0], mdata[0] - 1, std::less<T>());
	}
	else
	{
		T* d = (T*) dest.data_;
		T* s = (T*) srcs[0].data_;
		NElemT n = srcs[0].shape_.n_elems();
		NElemT out = 0;
		for (NElemT i = 1; i < n; ++i)
		{
			if (s[out] < s[i])
			{
				out = i;
			}
		}
		d[0] = out;
	}
}

template <typename T>
void reduce (NodeInfo dest, NodeInfo& src, DimT dim,
	std::function<void(T&,const T&)> accum)
{
	Shape& srcshape = src.shape_;
	T* d = (T*) dest.data_;
	T* s = (T*) src.data_;
	auto it = srcshape.begin();
	std::vector<DimT> front(it, it + dim);
	std::vector<DimT> back(it + dim + 1, srcshape.end());

	NElemT nouter = std::max<NElemT>(1, Shape(front).n_elems());
	NElemT ninner = srcshape.at(dim);
	NElemT nout2 = std::max<NElemT>(1, Shape(back).n_elems());

	for (NElemT i = 0; i < nouter; ++i)
	{
		for (NElemT j = 0; j < nout2; ++j)
		{
			NElemT outidx = i + j * nouter;
			NElemT inidx = i + j * ninner * nouter;
			NElemT n = i + (j + 1) * ninner * nouter;
			T out = s[inidx];
			for (inidx += nouter; inidx < n; inidx += nouter)
			{
				accum(out, s[inidx]);
			}
			d[outidx] = out;
		}
	}
}

template <typename T>
void rmax_helper (T& acc, const T& e)
{
	if (acc < e)
	{
		acc = e;
	}
}

template <typename T>
void reduce_max (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata)
{
	if (mdata[0] > 0)
	{
		reduce<T>(dest, srcs[0], mdata[0] - 1, rmax_helper<T>);
	}
	else
	{
		T* d = (T*) dest.data_;
		T* s = (T*) srcs[0].data_;
		NElemT n = srcs[0].shape_.n_elems();
		d[0] = s[0];
		for (NElemT i = 1; i < n; ++i)
		{
			rmax_helper<T>(d[0], s[i]);
		}
	}
}

template <typename T>
void rsum_helper (T& acc, const T& e)
{
	acc += e;
}

template <typename T>
void reduce_sum (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata)
{
	if (mdata[0] > 0)
	{
		reduce<T>(dest, srcs[0], mdata[0] - 1, rsum_helper<T>);
	}
	else
	{
		T* d = (T*) dest.data_;
		T* s = (T*) srcs[0].data_;
		NElemT n = srcs[0].shape_.n_elems();
		d[0] = s[0];
		for (NElemT i = 1; i < n; ++i)
		{
			rsum_helper<T>(d[0], s[i]);
		}
	}
}

}

#endif /* SAND_UNARY_HPP */
