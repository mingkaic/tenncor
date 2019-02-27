///
/// operator.hpp
/// ead
///
/// Purpose:
/// Define functions manipulating tensor data values
/// No function in this file makes any attempt to check for nullptrs
///

#include <unsupported/Eigen/MatrixFunctions>

#include "ead/tensor.hpp"
#include "ead/coord.hpp"
#include "ead/random.hpp"

#ifndef EAD_OPERATOR_HPP
#define EAD_OPERATOR_HPP

namespace ead
{

static inline bool is_2d (ade::Shape shape)
{
	return std::all_of(shape.begin() + 2, shape.end(),
		[](ade::DimT dim) { return 1 == dim; });
}

template <typename T>
struct OpArg final
{
	OpArg (T* data, ade::Shape shape, CoordMap* coorder) :
		data_(data), shape_(shape), coorder_(coorder) {}

	T* data_;

	ade::Shape shape_;

	CoordMap* coorder_ = nullptr;
};

template <typename OP, size_t N, typename T>
using ReduceOutT = Eigen::TensorReductionOp<OP,
	const std::array<ade::DimT,N>,const TensMapT<T>>;

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

#define _EAD_INTERNAL_V2A_CASE(N, PROCESS, RED)\
case N: return make_eigentensor<T,ReduceOutT<RED,N,T>,TensMapT<T>>(\
shape_convert(outshape), [vdims](TensMapT<T>& in) {\
return in.PROCESS(ead::internal::dim_copy<N>(vdims)); },\
make_tensmap(in.data_, in.shape_));

#define _EAD_INTERNAL_V2A(PROCESS, RED) {\
	assert(nullptr != in.coorder_);\
	ade::CoordT coord;\
	in.coorder_->forward(coord.begin(), coord.begin());\
	std::vector<ade::DimT> vdims;\
	std::copy_if(coord.begin(), coord.end(), std::back_inserter(vdims),\
		[](ade::DimT d) { return d < ade::rank_cap; });\
	switch (vdims.size()) {\
		_EAD_INTERNAL_V2A_CASE(0, PROCESS, RED)\
		_EAD_INTERNAL_V2A_CASE(1, PROCESS, RED)\
		_EAD_INTERNAL_V2A_CASE(2, PROCESS, RED)\
		_EAD_INTERNAL_V2A_CASE(3, PROCESS, RED)\
		_EAD_INTERNAL_V2A_CASE(4, PROCESS, RED)\
		_EAD_INTERNAL_V2A_CASE(5, PROCESS, RED)\
		_EAD_INTERNAL_V2A_CASE(6, PROCESS, RED)\
		_EAD_INTERNAL_V2A_CASE(7, PROCESS, RED)\
		default: break;\
	} return make_eigentensor<T,ReduceOutT<RED,8,T>,TensMapT<T>>(\
		shape_convert(outshape), [vdims](TensMapT<T>& in) {\
			return in.PROCESS(ead::internal::dim_copy<8>(vdims));\
		}, make_tensmap(in.data_, in.shape_));\
}

}

template <typename T>
EigenptrT<T> reduce_sum (ade::Shape& outshape, const OpArg<T>& in)
_EAD_INTERNAL_V2A(sum, Eigen::internal::SumReducer<T>)

template <typename T>
EigenptrT<T> reduce_prod (ade::Shape& outshape, const OpArg<T>& in)
_EAD_INTERNAL_V2A(prod, Eigen::internal::ProdReducer<T>)

template <typename T>
EigenptrT<T> reduce_min (ade::Shape& outshape, const OpArg<T>& in)
_EAD_INTERNAL_V2A(minimum, Eigen::internal::MinReducer<T>)

template <typename T>
EigenptrT<T> reduce_max (ade::Shape& outshape, const OpArg<T>& in)
_EAD_INTERNAL_V2A(maximum, Eigen::internal::MaxReducer<T>)

template <typename T>
EigenptrT<T> extend (ade::Shape& outshape, const OpArg<T>& in)
{
	assert(nullptr != in.coorder_);
	ade::CoordT coord;
	in.coorder_->forward(coord.begin(), coord.begin());
	return make_eigentensor<T,Eigen::TensorBroadcastingOp<
		const ade::CoordT,const TensMapT<T>>,TensMapT<T>>(
		shape_convert(outshape),
		[coord](TensMapT<T>& in)
		{
			return in.broadcast(coord);
		}, make_tensmap(in.data_, in.shape_));
}

template <typename T>
EigenptrT<T> permute (ade::Shape& outshape, const OpArg<T>& in)
{
	assert(nullptr != in.coorder_);
	ade::CoordT reorder;
	in.coorder_->forward(reorder.begin(), reorder.begin());
	return make_eigentensor<T,Eigen::TensorShufflingOp<
		const ade::CoordT,TensMapT<T>>,TensMapT<T>>(
		shape_convert(outshape),
		[reorder](TensMapT<T>& in)
		{
			return in.shuffle(reorder);
		}, make_tensmap(in.data_, in.shape_));
}

/// Given reference to output array, and input vector ref,
/// make output elements take absolute value of inputs
template <typename T>
EigenptrT<T> abs (ade::Shape& outshape, const OpArg<T>& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseUnaryOp<
			Eigen::internal::scalar_abs_op<T>,const MatMapT<T>>,
			MatMapT<T>>(shape_convert(outshape),
			[](MatMapT<T>& in)
			{
				return in.cwiseAbs();
			}, make_matmap(in.data_, in.shape_));
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_abs_op<T>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		[](TensMapT<T>& in)
		{
			return in.abs();
		}, make_tensmap(in.data_, in.shape_));
}

/// Given reference to output array, and input vector ref,
/// make output elements take negatives of inputs
template <typename T>
EigenptrT<T> neg (ade::Shape& outshape, const OpArg<T>& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseUnaryOp<
			Eigen::internal::scalar_opposite_op<T>,const MatMapT<T>>,
			MatMapT<T>>(shape_convert(outshape),
			[](MatMapT<T>& in)
			{
				return -in;
			}, make_matmap(in.data_, in.shape_));
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_opposite_op<T>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		[](TensMapT<T>& in)
		{
			return -in;
		}, make_tensmap(in.data_, in.shape_));
}

/// Given reference to output array, and input vector ref,
/// make output elements take sine of inputs
template <typename T>
EigenptrT<T> sin (ade::Shape& outshape, const OpArg<T>& in)
{
	if constexpr(!std::is_integral<T>::value)
	{
		if (is_2d(outshape))
		{
			// use matrix when possible
			return make_eigenmatrix<T,
				Eigen::MatrixFunctionReturnValue<MatMapT<T>>,
				MatMapT<T>>(shape_convert(outshape),
				[](MatMapT<T>& in)
				{
					return in.sin();
				}, make_matmap(in.data_, in.shape_));
		}
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		std::function<T(const T&)>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		[](TensMapT<T>& in)
		{
			return in.unaryExpr(std::function<T(const T&)>(
				[](const T& a) -> T
				{
					return std::sin(a);
				}));
		}, make_tensmap(in.data_, in.shape_));
}

/// Given reference to output array, and input vector ref,
/// make output elements take cosine of inputs
template <typename T>
EigenptrT<T> cos (ade::Shape& outshape, const OpArg<T>& in)
{
	if constexpr(!std::is_integral<T>::value)
	{
		if (is_2d(outshape))
		{
			// use matrix when possible
			return make_eigenmatrix<T,
				Eigen::MatrixFunctionReturnValue<MatMapT<T>>,
				MatMapT<T>>(shape_convert(outshape),
				[](MatMapT<T>& in)
				{
					return in.cos();
				}, make_matmap(in.data_, in.shape_));
		}
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		std::function<T(const T&)>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		[](TensMapT<T>& in)
		{
			return in.unaryExpr(std::function<T(const T&)>(
				[](const T& a) -> T
				{
					return std::cos(a);
				}));
		}, make_tensmap(in.data_, in.shape_));
}

/// Given reference to output array, and input vector ref,
/// make output elements take tangent of inputs
template <typename T>
EigenptrT<T> tan (ade::Shape& outshape, const OpArg<T>& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseUnaryOp<
			std::function<T(const T&)>,const MatMapT<T>>,
			MatMapT<T>>(shape_convert(outshape),
			[](MatMapT<T>& in)
			{
				return in.unaryExpr(std::function<T(const T&)>(
					[](const T& a) -> T
					{
						return std::tan(a);
					}));
			}, make_matmap(in.data_, in.shape_));
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		std::function<T(const T&)>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		[](TensMapT<T>& in)
		{
			return in.unaryExpr(std::function<T(const T&)>(
				[](const T& a) -> T
				{
					return std::tan(a);
				}));
		}, make_tensmap(in.data_, in.shape_));
}

/// Given reference to output array, and input vector ref,
/// make output elements take exponent of inputs
template <typename T>
EigenptrT<T> exp (ade::Shape& outshape, const OpArg<T>& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseUnaryOp<
			Eigen::internal::scalar_exp_op<T>,const MatMapT<T>>,
			MatMapT<T>>(shape_convert(outshape),
			[](MatMapT<T>& in)
			{
				return in.unaryExpr(Eigen::internal::scalar_exp_op<T>());
			}, make_matmap(in.data_, in.shape_));
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_exp_op<T>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		[](TensMapT<T>& in)
		{
			return in.exp();
		}, make_tensmap(in.data_, in.shape_));
}

/// Given reference to output array, and input vector ref,
/// make output elements take natural log of inputs
template <typename T>
EigenptrT<T> log (ade::Shape& outshape, const OpArg<T>& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseUnaryOp<
			Eigen::internal::scalar_log_op<T>,const MatMapT<T>>,
			MatMapT<T>>(shape_convert(outshape),
			[](MatMapT<T>& in)
			{
				return in.unaryExpr(Eigen::internal::scalar_log_op<T>());
			}, make_matmap(in.data_, in.shape_));
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_log_op<T>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		[](TensMapT<T>& in)
		{
			return in.log();
		}, make_tensmap(in.data_, in.shape_));
}

/// Given reference to output array, and input vector ref,
/// make output elements take square root of inputs
template <typename T>
EigenptrT<T> sqrt (ade::Shape& outshape, const OpArg<T>& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseUnaryOp<
			Eigen::internal::scalar_sqrt_op<T>,const MatMapT<T>>,
			MatMapT<T>>(shape_convert(outshape),
			[](MatMapT<T>& in)
			{
				return in.cwiseSqrt();
			}, make_matmap(in.data_, in.shape_));
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_sqrt_op<T>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		[](TensMapT<T>& in)
		{
			return in.sqrt();
		}, make_tensmap(in.data_, in.shape_));
}

/// Given reference to output array, and input vector ref,
/// make output elements take rounded values of inputs
template <typename T>
EigenptrT<T> round (ade::Shape& outshape, const OpArg<T>& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseUnaryOp<
			Eigen::internal::scalar_round_op<T>,const MatMapT<T>>,
			MatMapT<T>>(shape_convert(outshape),
			[](MatMapT<T>& in)
			{
				return in.unaryExpr(Eigen::internal::scalar_round_op<T>());
			}, make_matmap(in.data_, in.shape_));
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_round_op<T>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		[](TensMapT<T>& in)
		{
			return in.round();
		}, make_tensmap(in.data_, in.shape_));
}

template <typename T>
EigenptrT<T> sigmoid (ade::Shape& outshape, const OpArg<T>& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseUnaryOp<
			Eigen::internal::scalar_sigmoid_op<T>,const MatMapT<T>>,
			MatMapT<T>>(shape_convert(outshape),
			[](MatMapT<T>& in)
			{
				return in.unaryExpr(Eigen::internal::scalar_sigmoid_op<T>());
			}, make_matmap(in.data_, in.shape_));
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_sigmoid_op<T>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		[](TensMapT<T>& in)
		{
			return in.sigmoid();
		}, make_tensmap(in.data_, in.shape_));
}

template <typename T>
EigenptrT<T> tanh (ade::Shape& outshape, const OpArg<T>& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseUnaryOp<
			Eigen::internal::scalar_tanh_op<T>,const MatMapT<T>>,
			MatMapT<T>>(shape_convert(outshape),
			[](MatMapT<T>& in)
			{
				return in.unaryExpr(Eigen::internal::scalar_tanh_op<T>());
			}, make_matmap(in.data_, in.shape_));
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_tanh_op<T>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		[](TensMapT<T>& in)
		{
			return in.tanh();
		}, make_tensmap(in.data_, in.shape_));
}

template <typename T>
EigenptrT<T> square (ade::Shape& outshape, const OpArg<T>& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseUnaryOp<
			Eigen::internal::scalar_square_op<T>,const MatMapT<T>>,
			MatMapT<T>>(shape_convert(outshape),
			[](MatMapT<T>& in)
			{
				return in.unaryExpr(Eigen::internal::scalar_square_op<T>());
			}, make_matmap(in.data_, in.shape_));
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_square_op<T>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		[](TensMapT<T>& in)
		{
			return in.square();
		}, make_tensmap(in.data_, in.shape_));
}

template <typename T>
EigenptrT<T> cube (ade::Shape& outshape, const OpArg<T>& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseUnaryOp<
			Eigen::internal::scalar_cube_op<T>,const MatMapT<T>>,
			MatMapT<T>>(shape_convert(outshape),
			[](MatMapT<T>& in)
			{
				return in.unaryExpr(Eigen::internal::scalar_cube_op<T>());
			}, make_matmap(in.data_, in.shape_));
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_cube_op<T>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		[](TensMapT<T>& in)
		{
			return in.cube();
		}, make_tensmap(in.data_, in.shape_));
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply std::pow operator
/// Only accept 2 arguments
template <typename T>
EigenptrT<T> pow (ade::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseBinaryOp<
			std::function<T(const T&,const T&)>,
			const MatMapT<T>,const MatMapT<T>>,
			std::vector<MatMapT<T>>>(shape_convert(outshape),
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0].binaryExpr(args[1],
					std::function<T(const T&,const T&)>(
					[](const T& a, const T& b) -> T
					{
						return std::pow(a, b);
					}));
			}, {
				make_matmap(a.data_, a.shape_),
				make_matmap(b.data_, b.shape_)});
	}
	return make_eigentensor<T,Eigen::TensorCwiseBinaryOp<
		std::function<T(const T&,const T&)>,
		const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape),
		[](std::vector<TensMapT<T>>& args)
		{
			return args[0].binaryExpr(args[1],
				std::function<T(const T&,const T&)>(
				[](const T& a, const T& b) -> T
				{
					return std::pow(a, b);
				}));
		}, {
			make_tensmap(a.data_, a.shape_),
			make_tensmap(b.data_, b.shape_)});
}

template <typename T>
EigenptrT<T> add (ade::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseBinaryOp<
			Eigen::internal::scalar_sum_op<T>,
			const MatMapT<T>,const MatMapT<T>>,
			std::vector<MatMapT<T>>>(shape_convert(outshape),
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0] + args[1];
			}, {
				make_matmap(a.data_, a.shape_),
				make_matmap(b.data_, b.shape_)});
	}
	return make_eigentensor<T,Eigen::TensorCwiseBinaryOp<
		Eigen::internal::scalar_sum_op<T>,
		const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape),
		[](std::vector<TensMapT<T>>& args)
		{
			return args[0] + args[1];
		}, {
			make_tensmap(a.data_, a.shape_),
			make_tensmap(b.data_, b.shape_)});
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index subtract
/// Only accept 2 arguments
template <typename T>
EigenptrT<T> sub (ade::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseBinaryOp<
			Eigen::internal::scalar_difference_op<T>,
			const MatMapT<T>,const MatMapT<T>>,
			std::vector<MatMapT<T>>>(shape_convert(outshape),
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0] - args[1];
			}, {
				make_matmap(a.data_, a.shape_),
				make_matmap(b.data_, b.shape_)});
	}
	return make_eigentensor<T,Eigen::TensorCwiseBinaryOp<
		Eigen::internal::scalar_difference_op<T>,
		const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape),
		[](std::vector<TensMapT<T>>& args)
		{
			return args[0] - args[1];
		}, {
			make_tensmap(a.data_, a.shape_),
			make_tensmap(b.data_, b.shape_)});
}

template <typename T>
EigenptrT<T> mul (ade::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseBinaryOp<
			Eigen::internal::scalar_product_op<T>,
			const MatMapT<T>,const MatMapT<T>>,
			std::vector<MatMapT<T>>>(shape_convert(outshape),
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0].cwiseProduct(args[1]);
			}, {
				make_matmap(a.data_, a.shape_),
				make_matmap(b.data_, b.shape_)});
	}
	return make_eigentensor<T,Eigen::TensorCwiseBinaryOp<
		Eigen::internal::scalar_product_op<T>,
		const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape),
		[](std::vector<TensMapT<T>>& args)
		{
			return args[0] * args[1];
		}, {
			make_tensmap(a.data_, a.shape_),
			make_tensmap(b.data_, b.shape_)});
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index divide
/// Only accept 2 arguments
template <typename T>
EigenptrT<T> div (ade::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseBinaryOp<
			Eigen::internal::scalar_quotient_op<T>,
			const MatMapT<T>,const MatMapT<T>>,
			std::vector<MatMapT<T>>>(shape_convert(outshape),
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0].cwiseQuotient(args[1]);
			}, {
				make_matmap(a.data_, a.shape_),
				make_matmap(b.data_, b.shape_)});
	}
	return make_eigentensor<T,Eigen::TensorCwiseBinaryOp<
		Eigen::internal::scalar_quotient_op<T>,
		const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape),
		[](std::vector<TensMapT<T>>& args)
		{
			return args[0] / args[1];
		}, {
			make_tensmap(a.data_, a.shape_),
			make_tensmap(b.data_, b.shape_)});
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply == operator
/// Only accept 2 arguments
template <typename T>
EigenptrT<T> eq (ade::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseBinaryOp<
			std::function<T(const T&,const T&)>,
			const MatMapT<T>,const MatMapT<T>>,
			std::vector<MatMapT<T>>>(shape_convert(outshape),
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0].binaryExpr(args[1],
					std::function<T(const T&,const T&)>(
					[](const T& a, const T& b) -> T
					{
						return a == b;
					}));
			}, {
				make_matmap(a.data_, a.shape_),
				make_matmap(b.data_, b.shape_)});
	}
	return make_eigentensor<T,Eigen::TensorCwiseBinaryOp<
		std::function<T(const T&,const T&)>,
		const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape),
		[](std::vector<TensMapT<T>>& args)
		{
			return args[0].binaryExpr(args[1],
				std::function<T(const T&,const T&)>(
				[](const T& a, const T& b) -> T
				{
					return a == b;
				}));
		}, {
			make_tensmap(a.data_, a.shape_),
			make_tensmap(b.data_, b.shape_)});
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply != operator
/// Only accept 2 arguments
template <typename T>
EigenptrT<T> neq (ade::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseBinaryOp<
			std::function<T(const T&,const T&)>,
			const MatMapT<T>,const MatMapT<T>>,
			std::vector<MatMapT<T>>>(shape_convert(outshape),
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0].binaryExpr(args[1],
					std::function<T(const T&,const T&)>(
					[](const T& a, const T& b) -> T
					{
						return a != b;
					}));
			}, {
				make_matmap(a.data_, a.shape_),
				make_matmap(b.data_, b.shape_)});
	}
	return make_eigentensor<T,Eigen::TensorCwiseBinaryOp<
		std::function<T(const T&,const T&)>,
		const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape),
		[](std::vector<TensMapT<T>>& args)
		{
			return args[0].binaryExpr(args[1],
				std::function<T(const T&,const T&)>(
				[](const T& a, const T& b) -> T
				{
					return a != b;
				}));
		}, {
			make_tensmap(a.data_, a.shape_),
			make_tensmap(b.data_, b.shape_)});
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply < operator
/// Only accept 2 arguments
template <typename T>
EigenptrT<T> lt (ade::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseBinaryOp<
			std::function<T(const T&,const T&)>,
			const MatMapT<T>,const MatMapT<T>>,
			std::vector<MatMapT<T>>>(shape_convert(outshape),
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0].binaryExpr(args[1],
					std::function<T(const T&,const T&)>(
					[](const T& a, const T& b) -> T
					{
						return a < b;
					}));
			}, {
				make_matmap(a.data_, a.shape_),
				make_matmap(b.data_, b.shape_)});
	}
	return make_eigentensor<T,Eigen::TensorCwiseBinaryOp<
		std::function<T(const T&,const T&)>,
		const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape),
		[](std::vector<TensMapT<T>>& args)
		{
			return args[0].binaryExpr(args[1],
				std::function<T(const T&,const T&)>(
				[](const T& a, const T& b) -> T
				{
					return a < b;
				}));
		}, {
			make_tensmap(a.data_, a.shape_),
			make_tensmap(b.data_, b.shape_)});
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply > operator
/// Only accept 2 arguments
template <typename T>
EigenptrT<T> gt (ade::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseBinaryOp<
			std::function<T(const T&,const T&)>,
			const MatMapT<T>,const MatMapT<T>>,
			std::vector<MatMapT<T>>>(shape_convert(outshape),
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0].binaryExpr(args[1],
					std::function<T(const T&,const T&)>(
					[](const T& a, const T& b) -> T
					{
						return a > b;
					}));
			}, {
				make_matmap(a.data_, a.shape_),
				make_matmap(b.data_, b.shape_)});
	}
	return make_eigentensor<T,Eigen::TensorCwiseBinaryOp<
		std::function<T(const T&,const T&)>,
		const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape),
		[](std::vector<TensMapT<T>>& args)
		{
			return args[0].binaryExpr(args[1],
				std::function<T(const T&,const T&)>(
				[](const T& a, const T& b) -> T
				{
					return a > b;
				}));
		}, {
			make_tensmap(a.data_, a.shape_),
			make_tensmap(b.data_, b.shape_)});
}

/// Given arguments, for every mapped index i in range [0:max_nelems],
/// take the minimum all elements for all arguments
template <typename T>
EigenptrT<T> min (ade::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseBinaryOp<
			Eigen::internal::scalar_min_op<T,T>,
			const MatMapT<T>,const MatMapT<T>>,
			std::vector<MatMapT<T>>>(shape_convert(outshape),
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0].cwiseMin(args[1]);
			}, {
				make_matmap(a.data_, a.shape_),
				make_matmap(b.data_, b.shape_)});
	}
	return make_eigentensor<T,Eigen::TensorCwiseBinaryOp<
		Eigen::internal::scalar_min_op<T>,
		const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape),
		[](std::vector<TensMapT<T>>& args)
		{
			return args[0].cwiseMin(args[1]);
		}, {
			make_tensmap(a.data_, a.shape_),
			make_tensmap(b.data_, b.shape_)});
}

/// Given arguments, for every mapped index i in range [0:max_nelems],
/// take the maximum all elements for all arguments
template <typename T>
EigenptrT<T> max (ade::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseBinaryOp<
			Eigen::internal::scalar_max_op<T,T>,
			const MatMapT<T>,const MatMapT<T>>,
			std::vector<MatMapT<T>>>(shape_convert(outshape),
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0].cwiseMax(args[1]);
			}, {
				make_matmap(a.data_, a.shape_),
				make_matmap(b.data_, b.shape_)});
	}
	return make_eigentensor<T,Eigen::TensorCwiseBinaryOp<
		Eigen::internal::scalar_max_op<T>,
		const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape),
		[](std::vector<TensMapT<T>>& args)
		{
			return args[0].cwiseMax(args[1]);
		}, {
			make_tensmap(a.data_, a.shape_),
			make_tensmap(b.data_, b.shape_)});
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply std::uniform_distributon function
/// Only accept 2 arguments
template <typename T>
EigenptrT<T> rand_uniform (ade::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseBinaryOp<
			std::function<T(const T&,const T&)>,
			const MatMapT<T>,const MatMapT<T>>,
			std::vector<MatMapT<T>>>(shape_convert(outshape),
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0].binaryExpr(args[1],
					std::function<T(const T&,const T&)>(unif<T>));
			}, {
				make_matmap(a.data_, a.shape_),
				make_matmap(b.data_, b.shape_)});
	}
	return make_eigentensor<T,Eigen::TensorCwiseBinaryOp<
		std::function<T(const T&,const T&)>,
		const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape),
		[](std::vector<TensMapT<T>>& args)
		{
			return args[0].binaryExpr(args[1],
				std::function<T(const T&,const T&)>(unif<T>));
		}, {
			make_tensmap(a.data_, a.shape_),
			make_tensmap(b.data_, b.shape_)});
}

template <typename T>
EigenptrT<T> matmul (ade::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
{
	assert(is_2d(outshape));
	return make_eigenmatrix<T,Eigen::Product<MatMapT<T>,MatMapT<T>>,
		std::vector<MatMapT<T>>>(shape_convert(outshape),
		[](std::vector<MatMapT<T>>& args)
		{
			return args[0] * args[1];
		}, {
			make_matmap(a.data_, a.shape_),
			make_matmap(b.data_, b.shape_)});
}

template <typename T>
EigenptrT<T> convolution (ade::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
{
	throw std::bad_function_call(); // todo: implement
}

}

#endif // EAD_OPERATOR_HPP
