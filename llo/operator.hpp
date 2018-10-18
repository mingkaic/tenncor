///
/// operator.hpp
/// llo
///
/// Purpose:
/// Define functions manipulating tensor data values
/// No function in this file makes any attempt to check for nullptrs
///

#include <cstring>
#include <cmath>
#include <functional>
#include <random>

#include "ade/shape.hpp"

#ifndef LLO_OPERATOR_HPP
#define LLO_OPERATOR_HPP

namespace llo
{

/// RNG engine used
using EngineT = std::default_random_engine;

/// Return global random generator
EngineT& get_engine (void);

/// Tensor data wrapper using raw pointer and data size
/// Avoid using std constainers in case of unintentional deep copies
template <typename T>
struct VecRef
{
	const T* data;
	size_t n;
};

/// Generic unary operation
template <typename T>
void unary (T* out, VecRef<T> in, std::function<T(const T&)> f)
{
	for (size_t i = 0; i < in.n; ++i)
	{
		out[i] = f(in.data[i]);
	}
}

/// Given reference to output array, and input vector ref,
/// make output elements take absolute value of inputs
template <typename T>
void abs (T* out, VecRef<T> in)
{
	unary<T>(out, in, [](const T& src) { return std::abs(src); });
}

template <>
void abs<uint8_t> (uint8_t* out, VecRef<uint8_t> in);

template <>
void abs<uint16_t> (uint16_t* out, VecRef<uint16_t> in);

template <>
void abs<uint32_t> (uint32_t* out, VecRef<uint32_t> in);

template <>
void abs<uint64_t> (uint64_t* out, VecRef<uint64_t> in);

/// Given reference to output array, and input vector ref,
/// make output elements take negatives of inputs
template <typename T>
void neg (T* out, VecRef<T> in)
{
	unary<T>(out, in, [](const T& src) { return -src; });
}

template <>
void neg<uint8_t> (uint8_t* out, VecRef<uint8_t> in);

template <>
void neg<uint16_t> (uint16_t* out, VecRef<uint16_t> in);

template <>
void neg<uint32_t> (uint32_t* out, VecRef<uint32_t> in);

template <>
void neg<uint64_t> (uint64_t* out, VecRef<uint64_t> in);

/// Given reference to output array, and input vector ref,
/// make output elements take bitwise nots of inputs
template <typename T>
void bit_not (T* out, VecRef<T> in)
{
	unary<T>(out, in, [](const T& src) { return !src; });
}

/// Given reference to output array, and input vector ref,
/// make output elements take sine of inputs
template <typename T>
void sin (T* out, VecRef<T> in)
{
	unary<T>(out, in, [](const T& src) { return std::sin(src); });
}

/// Given reference to output array, and input vector ref,
/// make output elements take cosine of inputs
template <typename T>
void cos (T* out, VecRef<T> in)
{
	unary<T>(out, in, [](const T& src) { return std::cos(src); });
}

/// Given reference to output array, and input vector ref,
/// make output elements take tangent of inputs
template <typename T>
void tan (T* out, VecRef<T> in)
{
	unary<T>(out, in, [](const T& src) { return std::tan(src); });
}

/// Given reference to output array, and input vector ref,
/// make output elements take exponent of inputs
template <typename T>
void exp (T* out, VecRef<T> in)
{
	unary<T>(out, in, [](const T& src) { return std::exp(src); });
}

/// Given reference to output array, and input vector ref,
/// make output elements take natural log of inputs
template <typename T>
void log (T* out, VecRef<T> in)
{
	unary<T>(out, in, [](const T& src) { return std::log(src); });
}

/// Given reference to output array, and input vector ref,
/// make output elements take square root of inputs
template <typename T>
void sqrt (T* out, VecRef<T> in)
{
	unary<T>(out, in, [](const T& src) { return std::sqrt(src); });
}

/// Given reference to output array, and input vector ref,
/// make output elements take rounded values of inputs
template <typename T>
void round (T* out, VecRef<T> in)
{
	unary<T>(out, in, [](const T& src) { return std::round(src); });
}

/// Given reference to output array, and input vector ref,
/// and index of dimension to flip, map output and input elements such that
/// output.coordinate[dim] = shape[dim] - input.coordinate[dim]
/// From cartesian perspective, invert the order of values along an axis
/// For example: in=[[1, 2], [3, 4]], dim=1, yields out=[[3, 4], [1, 2]]
template <typename T>
void flip (T* out, const T* in, ade::Shape shape, uint8_t dim)
{
	size_t n = shape.n_elems();
	std::vector<ade::DimT> slist = shape.as_list();
	uint8_t rank = slist.size();
	if (dim >= rank)
	{
		ade::fatalf("attempting to flip dimension %d beyond shape rank %d",
			dim, rank);
	}
	std::vector<ade::DimT> coord;
	ade::DimT dlimit = slist[dim] - 1;
	for (size_t i = 0; i < n; ++i)
	{
		coord = coordinate(shape, i);
		coord[dim] = dlimit - coord[dim];
		out[i] = in[index(shape, coord)];
	}
}

/// Given a tensor argument and a vector of shape indices, permute tensor
/// The tensor's shape and each element's coordinates are reordered
/// according to shape indices
/// Unreferenced input shape dimensions are appended to the output shape
/// Input dimensions can be referenced more than once
/// Because output.nelems >= input.nelems, and coordinates are mapped 1-1,
/// Output indices that do not take input elements take on value 0
/// This 1-1 behavior is to facilitate creating identity matrices/tensors
template <typename T>
void permute (T* out, const T* in, ade::Shape outshape, ade::Shape shape,
	std::vector<uint8_t> order)
{
	size_t n = shape.n_elems();
	uint8_t orig_rank = shape.n_rank();
	bool visited[ade::rank_cap];
	std::memset(visited, false, sizeof(ade::DimT) * ade::rank_cap);
	for (size_t i = 0, n = order.size(); i < n; ++i)
	{
		visited[order[i]] = true;
	}
	for (size_t i = 0, n = orig_rank; i < n; ++i)
	{
		if (false == visited[i])
		{
			order.push_back(i);
		}
	}

	uint8_t norder = order.size();
	std::vector<ade::DimT> coords(orig_rank);
	std::vector<ade::DimT> converted(norder);
	std::memset(out, 0, sizeof(T) * outshape.n_elems());
	for (ade::NElemT srci = 0; srci < n; ++srci)
	{
		coords = coordinate(shape, srci);

		for (uint8_t i = 0; i < norder; ++i)
		{
			converted[i] = coords[order[i]];
		}

		ade::NElemT desti = index(outshape, converted);
		out[desti] = in[srci];
	}
}

/// Given a single argument get the n_elem value of the argument's shape
template <typename T>
void n_elems (T& out, const ade::Shape& in)
{
	out = in.n_elems();
}

/// Given an argument and a dimension index get the value of the argument's
/// shape at that index
template <typename T>
void n_dims (T& out, const ade::Shape& in, uint8_t dim)
{
	out = in.at(dim);
}

/// For each batch of in.n / nout elements,
/// assign first flat indices of the max value to out
/// Assert that in.n is a multiple of nout
template <typename T>
void arg_max (T* out, size_t nout, VecRef<T> in)
{
	size_t nbatch = in.n / nout;
	for (size_t i = 0; i < nout; ++i)
	{
		size_t temp = i * nbatch;
		for (size_t j = 1; j < nbatch; ++j)
		{
			if (in.data[temp] < in.data[i * nbatch + j])
			{
				temp = i * nbatch + j;
			}
		}
		out[i] = temp;
	}
}

/// For each batch of in.n / nout elements,
/// assign the max values to out
/// Assert that in.n is a multiple of nout
template <typename T>
void reduce_max (T* out, size_t nout, VecRef<T> in)
{
	size_t nbatch = in.n / nout;
	for (size_t i = 0; i < nout; ++i)
	{
		out[i] = in.data[i * nbatch];
		for (size_t j = 1; j < nbatch; ++j)
		{
			size_t k = i * nbatch + j;
			if (out[i] < in.data[k])
			{
				out[i] = in.data[k];
			}
		}
	}
}

/// For each batch of in.n / nout elements,
/// assign the sum of values to out
/// Assert that in.n is a multiple of nout
template <typename T>
void reduce_sum (T* out, size_t nout, VecRef<T> in)
{
	size_t nbatch = in.n / nout;
	for (size_t i = 0; i < nout; ++i)
	{
		out[i] = in.data[i * nbatch];
		for (size_t j = 1; j < nbatch; ++j)
		{
			out[i] += in.data[i * nbatch + j];
		}
	}
}

/// Generic binary operation
template <typename T>
void binary (T* out, VecRef<T> a, VecRef<T> b,
	std::function<T(const T&,const T&)> f)
{
	size_t n = std::max(a.n, b.n);
	for (size_t i = 0; i < n; ++i)
	{
		out[i] = f(a.data[i % a.n], b.data[i % b.n]);
	}
}

/// Generic n-nary operation
template <typename T>
void nnary (T* out, std::vector<VecRef<T>> args,
	std::function<void(T&, const T&)> acc)
{
	if (args.empty())
	{
		ade::fatal("Cannot perform operation with no arguments");
	}
	size_t n = std::max_element(args.begin(), args.end(),
	[](VecRef<T>& a, VecRef<T>& b)
	{
		return a.n < b.n;
	})->n;
	for (size_t i = 0; i < n; ++i)
	{
		out[i] = args[0].data[i % args[0].n];
		for (size_t j = 1, n = args.size(); j < n; ++j)
		{
			acc(out[i], args[j].data[i % args[j].n]);
		}
	}
}

/// Given arguments a, and b, for every index i in range [0:max_nelems],
/// apply std::pow operator to elements a[i % a.nelems] and b[i % b.nelems]
/// Shapes must be compatible before min_rank of both arguments
/// Only accept 2 arguments
template <typename T>
void pow (T* out, VecRef<T> a, VecRef<T> b)
{
	binary<T>(out, a, b,
		[](const T& b, const T& x) { return std::pow(b, x); });
}

/// Given arguments, for every index i in range [0:max_nelems],
/// sum all elements arg[i % arg.nelems] for arg in arguments
/// Shapes must be compatible before min_rank of all arguments
template <typename T>
void add (T* out, std::vector<VecRef<T>> args)
{
	nnary<T>(out, args, [](T& out, const T& val) { out += val; });
}

/// Given arguments a, and b, for every index i in range [0:max_nelems],
/// apply subtract elements a[i % a.nelems] and b[i % b.nelems]
/// Shapes must be compatible before min_rank of both arguments
/// Only accept 2 arguments
template <typename T>
void sub (T* out, VecRef<T> a, VecRef<T> b)
{
	binary<T>(out, a, b, [](const T& a, const T& b) { return a - b; });
}

/// Given arguments, for every index i in range [0:max_nelems],
/// multiply all elements arg[i % arg.nelems] for arg in arguments
/// Shapes must be compatible before min_rank of all arguments
template <typename T>
void mul (T* out, std::vector<VecRef<T>> args)
{
	nnary<T>(out, args, [](T& out, const T& val) { out *= val; });
}

/// Given arguments a, and b, for every index i in range [0:max_nelems],
/// apply divide elements a[i % a.nelems] and b[i % b.nelems]
/// Shapes must be compatible before min_rank of both arguments
/// Only accept 2 arguments
template <typename T>
void div (T* out, VecRef<T> a, VecRef<T> b)
{
	binary<T>(out, a, b, [](const T& a, const T& b) { return a / b; });
}

/// Given arguments a, and b, for every index i in range [0:max_nelems],
/// apply == operator to elements a[i % a.nelems] and b[i % b.nelems]
/// Shapes must be compatible before min_rank of both arguments
/// Only accept 2 arguments
template <typename T>
void eq (T* out, VecRef<T> a, VecRef<T> b)
{
	binary<T>(out, a, b, [](const T& a, const T& b) { return a == b; });
}

/// Given arguments a, and b, for every index i in range [0:max_nelems],
/// apply != operator to elements a[i % a.nelems] and b[i % b.nelems]
/// Shapes must be compatible before min_rank of both arguments
/// Only accept 2 arguments
template <typename T>
void neq (T* out, VecRef<T> a, VecRef<T> b)
{
	binary<T>(out, a, b, [](const T& a, const T& b) { return a != b; });
}

/// Given arguments a, and b, for every index i in range [0:max_nelems],
/// apply < operator to elements a[i % a.nelems] and b[i % b.nelems]
/// Shapes must be compatible before min_rank of both arguments
/// Only accept 2 arguments
template <typename T>
void lt (T* out, VecRef<T> a, VecRef<T> b)
{
	binary<T>(out, a, b, [](const T& a, const T& b) { return a < b; });
}

/// Given arguments a, and b, for every index i in range [0:max_nelems],
/// apply > operator to elements a[i % a.nelems] and b[i % b.nelems]
/// Shapes must be compatible before min_rank of both arguments
/// Only accept 2 arguments
template <typename T>
void gt (T* out, VecRef<T> a, VecRef<T> b)
{
	binary<T>(out, a, b, [](const T& a, const T& b) { return a > b; });
}

/// Given arguments, for every index i in range [0:max_nelems],
/// take the minimum all elements arg[i % arg.nelems]
/// for arg in arguments
/// Shapes must be compatible before min_rank of all arguments
template <typename T>
void min (T* out, std::vector<VecRef<T>> args)
{
	nnary<T>(out, args,
	[](T& out, const T& val) { out = std::min(out, val); });
}

/// Given arguments, for every index i in range [0:max_nelems],
/// take the maximum all elements arg[i % arg.nelems]
/// for arg in arguments
/// Shapes must be compatible before min_rank of all arguments
template <typename T>
void max (T* out, std::vector<VecRef<T>> args)
{
	nnary<T>(out, args,
	[](T& out, const T& val) { out = std::max(out, val); });
}

/// Given arguments a, and b, for every index i in range [0:max_nelems],
/// apply std::binomial_distribution function
/// to elements a[i % a.nelems] and b[i % b.nelems]
/// Shapes must be compatible before min_rank of both arguments
/// Only accept 2 arguments
template <typename T>
void rand_binom (T* out, VecRef<T> a, VecRef<double> b)
{
	size_t n = std::max(a.n, b.n);
	for (size_t i = 0; i < n; ++i)
	{
		std::binomial_distribution<T> dist(a.data[i % a.n], b.data[i % b.n]);
		out[i] = dist(get_engine());
	}
}

template <>
void rand_binom<double> (double* out, VecRef<double> a, VecRef<double> b);

template <>
void rand_binom<float> (float* out, VecRef<float> a, VecRef<double> b);

/// Given arguments a, and b, for every index i in range [0:max_nelems],
/// apply std::uniform_distributon function
/// to elements a[i % a.nelems] and b[i % b.nelems]
/// Shapes must be compatible before min_rank of both arguments
/// Only accept 2 arguments
template <typename T>
void rand_uniform (T* out, VecRef<T> a, VecRef<T> b)
{
	binary<T>(out, a, b,
	[](const T& a, const T& b)
	{
		std::uniform_int_distribution<T> dist(a, b);
		return dist(get_engine());
	});
}

template <>
void rand_uniform<double> (double* out, VecRef<double> a, VecRef<double> b);

template <>
void rand_uniform<float> (float* out, VecRef<float> a, VecRef<float> b);

/// Given arguments a, and b, for every index i in range [0:max_nelems],
/// apply std::normal_distribution function
/// to elements a[i % a.nelems] and b[i % b.nelems]
/// Shapes must be compatible before min_rank of both arguments
/// Only accept 2 arguments
template <typename T>
void rand_normal (T* out, VecRef<T> a, VecRef<T> b)
{
	throw std::bad_function_call();
}

template <>
void rand_normal<float> (float* out, VecRef<float> a, VecRef<float> b);

template <>
void rand_normal<double> (double* out, VecRef<double> a, VecRef<double> b);

/// Given 2 arguments, matrix multiply
/// The # of column of the first argument must match the nrow of the second
/// Given the arguments and 2 indices, for each argument
/// form groups [:idx) and [index:rank) and treat dimensions falling in
/// those ranges as a single dimension (where the shape values must match)
/// then apply matmul given the grouped shape
/// For example, given shapea={3, 4, 5}, ai=2, shapeb={7, 8, 3, 4}, bi=2,
/// output tensor has shape {7, 8, 5}, since {3, 4} in a and b matches
template <typename T>
void matmul (T* out, const T* a, const T* b,
	const ade::Shape& ashape, const ade::Shape& bshape,
	uint8_t agroup_idx, uint8_t bgroup_idx)
{
	auto ita = ashape.begin();
	auto itb = bshape.begin();
	ade::NElemT dim_x = std::accumulate(itb, itb + bgroup_idx,
		1, std::multiplies<ade::NElemT>());
	ade::NElemT dim_y = std::accumulate(
		ita + agroup_idx, ita + ashape.n_rank(),
		1, std::multiplies<ade::NElemT>());
	ade::NElemT dim_z = std::accumulate(ita, ita + agroup_idx,
		1, std::multiplies<ade::NElemT>());

	for (size_t y = 0; y < dim_y; y++)
	{
		for (size_t x = 0; x < dim_x; x++)
		{
			size_t outidx = x + y * dim_x;
			out[outidx] = 0;
			for (size_t z = 0; z < dim_z; z++)
			{
				size_t aidx = dim_z * y + z;
				size_t bidx = x + dim_x * z;
				out[outidx] += a[aidx] * b[bidx];
			}
		}
	}
}

/// Given a single input ref and output information
/// (including length of output vector), copy over data
/// repeating input data to fit when necessary.
template <typename T>
void copyover (T* out, size_t nout, VecRef<T> in)
{
	size_t mult = nout / in.n;
	for (size_t i = 0; i < mult; ++i)
	{
		std::memcpy(out + i * in.n, in.data, sizeof(T) * in.n);
	}
	if (size_t leftover = nout % in.n)
	{
		std::memcpy(out + mult * in.n, in.data, sizeof(T) * leftover);
	}
}

}

#endif // LLO_OPERATOR_HPP
