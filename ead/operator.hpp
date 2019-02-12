///
/// operator.hpp
/// ead
///
/// Purpose:
/// Define functions manipulating tensor data values
/// No function in this file makes any attempt to check for nullptrs
///

#include "ead/tensor.hpp"
#include "ead/coord.hpp"

#ifndef EAD_OPERATOR_HPP
#define EAD_OPERATOR_HPP

namespace ead
{

template <typename T>
struct OpArg final
{
	OpArg (ade::Shape shape, TensMapT<T>* tensmap, CoordMap* coorder) :
		shape_(shape), tensmap_(tensmap), coorder_(coorder) {}

	ade::Shape shape_;

	TensMapT<T>* tensmap_;

	CoordMap* coorder_ = nullptr;
};

/// RNG engine used
using EngineT = std::default_random_engine;

/// Return global random generator
EngineT& get_engine (void);

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply std::uniform_distributon function
/// Only accept 2 arguments
template <typename T>
EigenptrT<T> rand_uniform (ade::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
{
	return make_tensop<T>(outshape, a.tensmap_->binaryExpr(*b.tensmap_,
		[](const ScalarT<T>& a, const ScalarT<T>& b)
		{
			std::uniform_int_distribution<T> dist(a, b);
			return dist(get_engine());
		}));
}

template <>
EigenptrT<double> rand_uniform<double> (ade::Shape& outshape, const OpArg<double>& a, const OpArg<double>& b);

template <>
EigenptrT<float> rand_uniform<float> (ade::Shape& outshape, const OpArg<float>& a, const OpArg<float>& b);

namespace internal
{

template <size_t N>
inline std::array<ade::DimT,N> dim_copy (std::vector<ade::DimT> d)
{
	std::array<ade::DimT,N> out;
	auto it = d.begin();
	std::copy(it, it + N, out.begin());
	return out;
}

#define _EAD_INTERNAL_V2A_CASE(N, PROCESS)\
case N: return make_tensop<T>(outshape,\
in.tensmap_->PROCESS(::ead::internal::dim_copy<N>(vdims)));

#define _EAD_INTERNAL_V2A(PROCESS) {\
	assert(nullptr != in.coorder_);\
	ade::CoordT coord;\
	in.coorder_->forward(coord.begin(), coord.begin());\
	std::vector<ade::DimT> vdims;\
	std::copy_if(coord.begin(), coord.end(), std::back_inserter(vdims),\
		[](ade::DimT d) { return d < ade::rank_cap; });\
	switch (vdims.size()) {\
		_EAD_INTERNAL_V2A_CASE(1, PROCESS)\
		_EAD_INTERNAL_V2A_CASE(2, PROCESS)\
		_EAD_INTERNAL_V2A_CASE(3, PROCESS)\
		_EAD_INTERNAL_V2A_CASE(4, PROCESS)\
		_EAD_INTERNAL_V2A_CASE(5, PROCESS)\
		_EAD_INTERNAL_V2A_CASE(6, PROCESS)\
		_EAD_INTERNAL_V2A_CASE(7, PROCESS)\
		default: break;\
	} return make_tensop<T>(outshape,\
		in.tensmap_->PROCESS(::ead::internal::dim_copy<8>(vdims)));\
}

}

template <typename T>
EigenptrT<T> reduce_sum (ade::Shape& outshape, const OpArg<T>& in)
_EAD_INTERNAL_V2A(sum)

template <typename T>
EigenptrT<T> reduce_prod (ade::Shape& outshape, const OpArg<T>& in)
_EAD_INTERNAL_V2A(prod)

template <typename T>
EigenptrT<T> reduce_min (ade::Shape& outshape, const OpArg<T>& in)
_EAD_INTERNAL_V2A(minimum)

template <typename T>
EigenptrT<T> reduce_max (ade::Shape& outshape, const OpArg<T>& in)
_EAD_INTERNAL_V2A(maximum)

template <typename T>
EigenptrT<T> extend (ade::Shape& outshape, const OpArg<T>& in)
{
	assert(nullptr != in.coorder_);
	ade::CoordT coord;
	in.coorder_->forward(coord.begin(), coord.begin());
	return make_tensop<T>(outshape, in.tensmap_->broadcast(coord));
}

template <typename T>
EigenptrT<T> permute (ade::Shape& outshape, const OpArg<T>& in)
{
	assert(nullptr != in.coorder_);
	ade::CoordT reorder;
	in.coorder_->forward(reorder.begin(), reorder.begin());
	return make_tensop<T>(outshape, in.tensmap_->shuffle(reorder));
}

/// Given reference to output array, and input vector ref,
/// make output elements take absolute value of inputs
template <typename T>
EigenptrT<T> abs (ade::Shape& outshape, const OpArg<T>& in)
{
	return make_tensop<T>(outshape, in.tensmap_->abs());
}

/// Given reference to output array, and input vector ref,
/// make output elements take negatives of inputs
template <typename T>
EigenptrT<T> neg (ade::Shape& outshape, const OpArg<T>& in)
{
	return make_tensop<T>(outshape, -(*in.tensmap_));
}

/// Given reference to output array, and input vector ref,
/// make output elements take sine of inputs
template <typename T>
EigenptrT<T> sin (ade::Shape& outshape, const OpArg<T>& in)
{
	logs::fatal("sine function is disabled for integral types");
}

template <>
EigenptrT<double> sin<double> (ade::Shape& outshape, const OpArg<double>& in);

template <>
EigenptrT<float> sin<float> (ade::Shape& outshape, const OpArg<float>& in);

/// Given reference to output array, and input vector ref,
/// make output elements take cosine of inputs
template <typename T>
EigenptrT<T> cos (ade::Shape& outshape, const OpArg<T>& in)
{
	logs::fatal("cosine function is disabled for integral types");
}

template <>
EigenptrT<double> cos<double> (ade::Shape& outshape, const OpArg<double>& in);

template <>
EigenptrT<float> cos<float> (ade::Shape& outshape, const OpArg<float>& in);

/// Given reference to output array, and input vector ref,
/// make output elements take tangent of inputs
template <typename T>
EigenptrT<T> tan (ade::Shape& outshape, const OpArg<T>& in)
{
	logs::fatal("tangent function is disabled for integral types");
}

template <>
EigenptrT<double> tan<double> (ade::Shape& outshape, const OpArg<double>& in);

template <>
EigenptrT<float> tan<float> (ade::Shape& outshape, const OpArg<float>& in);

/// Given reference to output array, and input vector ref,
/// make output elements take exponent of inputs
template <typename T>
EigenptrT<T> exp (ade::Shape& outshape, const OpArg<T>& in)
{
	return make_tensop<T>(outshape, in.tensmap_->exp());
}

/// Given reference to output array, and input vector ref,
/// make output elements take natural log of inputs
template <typename T>
EigenptrT<T> log (ade::Shape& outshape, const OpArg<T>& in)
{
	return make_tensop<T>(outshape, in.tensmap_->log());
}

/// Given reference to output array, and input vector ref,
/// make output elements take square root of inputs
template <typename T>
EigenptrT<T> sqrt (ade::Shape& outshape, const OpArg<T>& in)
{
	return make_tensop<T>(outshape, in.tensmap_->sqrt());
}

/// Given reference to output array, and input vector ref,
/// make output elements take rounded values of inputs
template <typename T>
EigenptrT<T> round (ade::Shape& outshape, const OpArg<T>& in)
{
	return make_tensop<T>(outshape, in.tensmap_->round());
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply std::pow operator
/// Only accept 2 arguments
template <typename T>
EigenptrT<T> pow (ade::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
{
	return make_tensop<T>(outshape, a.tensmap_->binaryExpr(*b.tensmap_,
		[](const ScalarT<T>& a, const ScalarT<T>& b) -> ScalarT<T>
		{
			return std::pow(a, b);
		}));
}

template <typename T>
EigenptrT<T> add (ade::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
{
	return make_tensop<T>(outshape, *a.tensmap_ + *b.tensmap_);
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index subtract
/// Only accept 2 arguments
template <typename T>
EigenptrT<T> sub (ade::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
{
	return make_tensop<T>(outshape, *a.tensmap_ - *b.tensmap_);
}

template <typename T>
EigenptrT<T> mul (ade::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
{
	return make_tensop<T>(outshape, *a.tensmap_ * *b.tensmap_);
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index divide
/// Only accept 2 arguments
template <typename T>
EigenptrT<T> div (ade::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
{
	return make_tensop<T>(outshape, *a.tensmap_ / *b.tensmap_);
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply == operator
/// Only accept 2 arguments
template <typename T>
EigenptrT<T> eq (ade::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
{
	return make_tensop<T>(outshape, a.tensmap_->binaryExpr(*b.tensmap_,
		[](const ScalarT<T>& a, const ScalarT<T>& b) -> ScalarT<T>
		{
			return a == b;
		}));
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply != operator
/// Only accept 2 arguments
template <typename T>
EigenptrT<T> neq (ade::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
{
	return make_tensop<T>(outshape, a.tensmap_->binaryExpr(*b.tensmap_,
		[](const ScalarT<T>& a, const ScalarT<T>& b) -> ScalarT<T>
		{
			return a != b;
		}));
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply < operator
/// Only accept 2 arguments
template <typename T>
EigenptrT<T> lt (ade::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
{
	return make_tensop<T>(outshape, a.tensmap_->binaryExpr(*b.tensmap_,
		[](const ScalarT<T>& a, const ScalarT<T>& b) -> ScalarT<T>
		{
			return a < b;
		}));
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply > operator
/// Only accept 2 arguments
template <typename T>
EigenptrT<T> gt (ade::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
{
	return make_tensop<T>(outshape, a.tensmap_->binaryExpr(*b.tensmap_,
		[](const ScalarT<T>& a, const ScalarT<T>& b) -> ScalarT<T>
		{
			return a > b;
		}));
}

/// Given arguments, for every mapped index i in range [0:max_nelems],
/// take the minimum all elements for all arguments
template <typename T>
EigenptrT<T> min (ade::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
{
	return make_tensop<T>(outshape, a.tensmap_->cwiseMin(*b.tensmap_));
}

/// Given arguments, for every mapped index i in range [0:max_nelems],
/// take the maximum all elements for all arguments
template <typename T>
EigenptrT<T> max (ade::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
{
	return make_tensop<T>(outshape, a.tensmap_->cwiseMax(*b.tensmap_));
}

template <typename T>
EigenptrT<T> matmul (ade::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
{
	return make_matop<T>(outshape,
		tensmap_to_matmap(*a.tensmap_) * tensmap_to_matmap(*b.tensmap_));
}

template <typename T>
EigenptrT<T> convolution (ade::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
{
	throw std::bad_function_call(); // todo: implement
}

// sigmoid

}

#endif // EAD_OPERATOR_HPP
