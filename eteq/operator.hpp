///
/// operator.hpp
/// eteq
///
/// Purpose:
/// Define functions manipulating tensor data values
/// No function in this file makes any attempt to check for nullptrs
///

#include "eteq/eigen.hpp"
#include "eteq/coord.hpp"
#include "eteq/random.hpp"

#ifndef ETEQ_OPERATOR_HPP
#define ETEQ_OPERATOR_HPP

namespace eteq
{

static inline bool is_2d (teq::Shape shape)
{
	return std::all_of(shape.begin() + 2, shape.end(),
		[](teq::DimT dim) { return 1 == dim; });
}

/// Raw data, shape, and transformation argument struct
template <typename T>
struct OpArg final
{
	OpArg (T* data, teq::Shape shape, CoordMap* coorder) :
		data_(data), shape_(shape), coorder_(coorder) {}

	/// Raw data argument
	T* data_;

	/// Shape of the data
	teq::Shape shape_;

	/// Transformation argument, null denotes no argument
	CoordMap* coorder_ = nullptr;
};

/// Generic Eigen reduction operator
template <typename OP, size_t N, typename T>
using ReduceOutT = Eigen::TensorReductionOp<OP,
	const std::array<teq::RankT,N>,const TensMapT<T>>;

namespace internal
{

/// Return array of input vector
template <size_t N>
inline std::array<teq::RankT,N> dim_copy (std::vector<teq::RankT> d)
{
	std::array<teq::RankT,N> out;
	auto it = d.begin();
	std::copy(it, it + N, out.begin());
	return out;
}

#define _ETEQ_INTERNAL_V2A_CASE(N, PROCESS, RED)\
case N: return make_eigentensor<T,ReduceOutT<RED,N,T>,TensMapT<T>>(\
shape_convert(outshape), [vdims](TensMapT<T>& in) {\
return in.PROCESS(::eteq::internal::dim_copy<N>(vdims)); },\
make_tensmap(in.data_, in.shape_));

#define _ETEQ_INTERNAL_V2A(PROCESS, RED) {\
	assert(nullptr != in.coorder_);\
	std::vector<teq::RankT> vdims;\
	vdims.reserve(teq::rank_cap);\
	in.coorder_->access([&](const teq::MatrixT& args){\
		for (teq::RankT i = 0; i < teq::rank_cap &&\
			args[0][i] < teq::rank_cap; ++i)\
		{ vdims.push_back(args[0][i]); }\
	});\
	switch (vdims.size()) {\
		_ETEQ_INTERNAL_V2A_CASE(0, PROCESS, RED)\
		_ETEQ_INTERNAL_V2A_CASE(1, PROCESS, RED)\
		_ETEQ_INTERNAL_V2A_CASE(2, PROCESS, RED)\
		_ETEQ_INTERNAL_V2A_CASE(3, PROCESS, RED)\
		_ETEQ_INTERNAL_V2A_CASE(4, PROCESS, RED)\
		_ETEQ_INTERNAL_V2A_CASE(5, PROCESS, RED)\
		_ETEQ_INTERNAL_V2A_CASE(6, PROCESS, RED)\
		_ETEQ_INTERNAL_V2A_CASE(7, PROCESS, RED)\
		default: break;\
	} return make_eigentensor<T,ReduceOutT<RED,8,T>,TensMapT<T>>(\
		shape_convert(outshape), [vdims](TensMapT<T>& in) {\
			return in.PROCESS(::eteq::internal::dim_copy<8>(vdims));\
		}, make_tensmap(in.data_, in.shape_));\
}

}

/// Return Eigen data object representing reduction where aggregation is sum
template <typename T>
EigenptrT<T> reduce_sum (teq::Shape& outshape, const OpArg<T>& in)
_ETEQ_INTERNAL_V2A(sum, Eigen::internal::SumReducer<T>)

/// Return Eigen data object representing reduction where aggregation is prod
template <typename T>
EigenptrT<T> reduce_prod (teq::Shape& outshape, const OpArg<T>& in)
_ETEQ_INTERNAL_V2A(prod, Eigen::internal::ProdReducer<T>)

/// Return Eigen data object representing reduction where aggregation is min
template <typename T>
EigenptrT<T> reduce_min (teq::Shape& outshape, const OpArg<T>& in)
_ETEQ_INTERNAL_V2A(minimum, Eigen::internal::MinReducer<T>)

/// Return Eigen data object representing reduction where aggregation is max
template <typename T>
EigenptrT<T> reduce_max (teq::Shape& outshape, const OpArg<T>& in)
_ETEQ_INTERNAL_V2A(maximum, Eigen::internal::MaxReducer<T>)

/// Return Eigen data object representing data broadcast across dimensions
template <typename T>
EigenptrT<T> extend (teq::Shape& outshape, const OpArg<T>& in)
{
	assert(nullptr != in.coorder_);
	teq::CoordT coord;
	in.coorder_->access(
		[&](const teq::MatrixT& args)
		{
			for (teq::RankT i = 0; i < teq::rank_cap; ++i)
			{
				coord[i] = args[0][i];
			}
		});
	return make_eigentensor<T,Eigen::TensorBroadcastingOp<
		const teq::CoordT,const TensMapT<T>>,TensMapT<T>>(
		shape_convert(outshape),
		[coord](TensMapT<T>& in)
		{
			return in.broadcast(coord);
		}, make_tensmap(in.data_, in.shape_));
}

/// Return Eigen data object representing transpose and permutation
template <typename T>
EigenptrT<T> permute (teq::Shape& outshape, const OpArg<T>& in)
{
	assert(nullptr != in.coorder_);
	teq::CoordT reorder;
	in.coorder_->access(
		[&](const teq::MatrixT& args)
		{
			for (teq::RankT i = 0; i < teq::rank_cap; ++i)
			{
				reorder[i] = args[0][i];
			}
		});
	if (is_2d(outshape) && reorder[0] == 1 && reorder[1] == 0)
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::Transpose<MatMapT<T>>,
			MatMapT<T>>(
			shape_convert(outshape),
			[](MatMapT<T>& in)
			{
				return in.transpose();
			}, make_matmap(in.data_, in.shape_));
	}
	return make_eigentensor<T,Eigen::TensorShufflingOp<
		const teq::CoordT,TensMapT<T>>,TensMapT<T>>(
		shape_convert(outshape),
		[reorder](TensMapT<T>& in)
		{
			return in.shuffle(reorder);
		}, make_tensmap(in.data_, in.shape_));
}

/// Return Eigen data object representing data slicing of dimensions
template <typename T>
EigenptrT<T> slice (teq::Shape& outshape, const OpArg<T>& in)
{
	assert(nullptr != in.coorder_);
	teq::ShapeT offsets;
	teq::ShapeT extents;
	in.coorder_->access(
		[&](const teq::MatrixT& args)
		{
			std::copy(args[0], args[0] + teq::rank_cap, offsets.begin());
			std::copy(args[1], args[1] + teq::rank_cap, extents.begin());
		});
	return make_eigentensor<T,Eigen::TensorSlicingOp<
			const teq::ShapeT, const teq::ShapeT,
			TensMapT<T>
		>,
		TensMapT<T>>(
		shape_convert(outshape),
		[&offsets, &extents](TensMapT<T>& in)
		{
			return in.slice(offsets, extents);
		}, make_tensmap(in.data_, in.shape_));
}

/// Return Eigen data object representing data zero padding
template <typename T>
EigenptrT<T> pad (teq::Shape& outshape, const OpArg<T>& in)
{
	assert(nullptr != in.coorder_);
	std::array<std::pair<teq::DimT,teq::DimT>,teq::rank_cap> paddings;
	in.coorder_->access(
		[&](const teq::MatrixT& args)
		{
			for (teq::RankT i = 0; i < teq::rank_cap; ++i)
			{
				paddings[i] = {args[0][i], args[1][i]};
			}
		});
	return make_eigentensor<T,Eigen::TensorPaddingOp<
			const std::array<std::pair<teq::DimT,teq::DimT>,teq::rank_cap>,
			const TensMapT<T>
		>,
		TensMapT<T>>(
		shape_convert(outshape),
		[&paddings](TensMapT<T>& in)
		{
			return in.pad(paddings);
		}, make_tensmap(in.data_, in.shape_));
}

/// Return Eigen data object representing strided view of in
template <typename T>
EigenptrT<T> stride (teq::Shape& outshape, const OpArg<T>& in)
{
	assert(nullptr != in.coorder_);
	Eigen::array<Eigen::DenseIndex,teq::rank_cap> incrs;
	in.coorder_->access(
		[&](const teq::MatrixT& args)
		{
			for (teq::RankT i = 0; i < teq::rank_cap; ++i)
			{
				incrs[i] = args[0][i];
			}
		});
	return make_eigentensor<T,Eigen::TensorStridingOp<
			const Eigen::array<Eigen::DenseIndex,teq::rank_cap>,
			TensMapT<T>
		>,
		TensMapT<T>>(
		shape_convert(outshape),
		[&incrs](TensMapT<T>& in)
		{
			return in.stride(incrs);
		}, make_tensmap(in.data_, in.shape_));
}

/// Return Eigen data object that scatters data in
/// specific increments across dimensions
/// This function is the reverse of stride
template <typename T>
EigenptrT<T> scatter (teq::Shape& outshape, const OpArg<T>& in)
{
	assert(nullptr != in.coorder_);
	Eigen::array<Eigen::DenseIndex,teq::rank_cap> incrs;
	in.coorder_->access(
		[&](const teq::MatrixT& args)
		{
			for (teq::RankT i = 0; i < teq::rank_cap; ++i)
			{
				incrs[i] = args[0][i];
			}
		});
	return std::make_shared<EigenAssignTens<T>>(shape_convert(outshape),
		make_tensmap(in.data_, in.shape_),
		[incrs](TensorT<T>& out, const TensMapT<T>& in)
		{
			out.stride(incrs) = in;
		});
}

/// Given reference to output array, and input vector ref,
/// make output elements take absolute value of inputs
template <typename T>
EigenptrT<T> abs (teq::Shape& outshape, const OpArg<T>& in)
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
EigenptrT<T> neg (teq::Shape& outshape, const OpArg<T>& in)
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
EigenptrT<T> sin (teq::Shape& outshape, const OpArg<T>& in)
{
#ifdef __cpp_if_constexpr
	if constexpr(!std::is_integral<T>::value)
	{
		if (is_2d(outshape))
		{
			// use matrix when possible
			return make_eigenmatrix<T,
				typename Eigen::ArrayWrapper<MatMapT<T>>::SinReturnType,
				MatMapT<T>>(shape_convert(outshape),
				[](MatMapT<T>& in)
				{
					return in.array().sin();
				}, make_matmap(in.data_, in.shape_));
		}
	}
#endif
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
EigenptrT<T> cos (teq::Shape& outshape, const OpArg<T>& in)
{
#ifdef __cpp_if_constexpr
	if constexpr(!std::is_integral<T>::value)
	{
		if (is_2d(outshape))
		{
			// use matrix when possible
			return make_eigenmatrix<T,
				typename Eigen::ArrayWrapper<MatMapT<T>>::CosReturnType,
				MatMapT<T>>(shape_convert(outshape),
				[](MatMapT<T>& in)
				{
					return in.array().cos();
				}, make_matmap(in.data_, in.shape_));
		}
	}
#endif
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
EigenptrT<T> tan (teq::Shape& outshape, const OpArg<T>& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,
			typename Eigen::ArrayWrapper<MatMapT<T>>::TanReturnType,
			MatMapT<T>>(shape_convert(outshape),
			[](MatMapT<T>& in)
			{
				return in.array().tan();
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
EigenptrT<T> exp (teq::Shape& outshape, const OpArg<T>& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,
			typename Eigen::ArrayWrapper<MatMapT<T>>::ExpReturnType,
			MatMapT<T>>(shape_convert(outshape),
			[](MatMapT<T>& in)
			{
				return in.array().exp();
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
EigenptrT<T> log (teq::Shape& outshape, const OpArg<T>& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,
			typename Eigen::ArrayWrapper<MatMapT<T>>::LogReturnType,
			MatMapT<T>>(shape_convert(outshape),
			[](MatMapT<T>& in)
			{
				return in.array().log();
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
EigenptrT<T> sqrt (teq::Shape& outshape, const OpArg<T>& in)
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
EigenptrT<T> round (teq::Shape& outshape, const OpArg<T>& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,
			typename Eigen::ArrayWrapper<MatMapT<T>>::RoundReturnType,
			MatMapT<T>>(shape_convert(outshape),
			[](MatMapT<T>& in)
			{
				return in.array().round();
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
EigenptrT<T> sigmoid (teq::Shape& outshape, const OpArg<T>& in)
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
EigenptrT<T> sigmoid_grad (teq::Shape& outshape, const OpArg<T>& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,
			Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<T>,
				const Eigen::CwiseUnaryOp<Eigen::internal::scalar_sigmoid_op<T>,
					const MatMapT<T>>,
				const Eigen::CwiseUnaryOp<Eigen::internal::bind1st_op<
					Eigen::internal::scalar_difference_op<T>>,
					const Eigen::CwiseUnaryOp<Eigen::internal::scalar_sigmoid_op<T>,
						const MatMapT<T>>>>,
			MatMapT<T>>(shape_convert(outshape),
			[](MatMapT<T>& in)
			{
				auto out = in.unaryExpr(Eigen::internal::scalar_sigmoid_op<T>());
				return out.cwiseProduct(out.unaryExpr(
					Eigen::internal::bind1st_op<Eigen::internal::scalar_difference_op<T>>(1)));
			}, make_matmap(in.data_, in.shape_));
	}
	return make_eigentensor<T,
		Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<T>,
			const Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_sigmoid_op<T>,
				const TensMapT<T>>,
			const Eigen::TensorCwiseUnaryOp<Eigen::internal::bind1st_op<
				Eigen::internal::scalar_difference_op<T>>,
				const Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_sigmoid_op<T>,
					const TensMapT<T>>>>,
		TensMapT<T>>(shape_convert(outshape),
		[](TensMapT<T>& in)
		{
			auto out = in.sigmoid();
			return out * (1 - out);
		}, make_tensmap(in.data_, in.shape_));
}

template <typename T>
EigenptrT<T> tanh (teq::Shape& outshape, const OpArg<T>& in)
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
EigenptrT<T> square (teq::Shape& outshape, const OpArg<T>& in)
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
EigenptrT<T> cube (teq::Shape& outshape, const OpArg<T>& in)
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
EigenptrT<T> pow (teq::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
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
EigenptrT<T> add (teq::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
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
EigenptrT<T> sub (teq::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
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
EigenptrT<T> mul (teq::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
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
EigenptrT<T> div (teq::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
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
EigenptrT<T> eq (teq::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
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
EigenptrT<T> neq (teq::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
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
EigenptrT<T> lt (teq::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
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
EigenptrT<T> gt (teq::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
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
EigenptrT<T> min (teq::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
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
EigenptrT<T> max (teq::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
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
EigenptrT<T> rand_uniform (teq::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
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

/// Given a condition, then values and otherwise
/// apply corresponding then value if condition is non-zero
/// otherwise apply otherwise value
template <typename T>
EigenptrT<T> select (teq::Shape& outshape,
	const OpArg<T>& condition,
	const OpArg<T>& then, const OpArg<T>& otherwise)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,
			Eigen::Select<MatMapT<T>,MatMapT<T>,MatMapT<T>>,
			std::vector<MatMapT<T>>>(shape_convert(outshape),
			[](std::vector<MatMapT<T>>& args) -> Eigen::Select<MatMapT<T>,MatMapT<T>,MatMapT<T>>
			{
				return args[0].select(args[1], args[2]);
			}, {
				make_matmap(condition.data_, condition.shape_),
				make_matmap(then.data_, then.shape_),
				make_matmap(otherwise.data_, otherwise.shape_)});
	}
	return make_eigentensor<T,
		Eigen::TensorSelectOp<const TensMapT<T>,
			const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape),
		[](std::vector<TensMapT<T>>& args) -> Eigen::TensorSelectOp<const TensMapT<T>,const TensMapT<T>,const TensMapT<T>>
		{
			return args[0].select(args[1], args[2]);
		}, {
			make_tensmap(condition.data_, condition.shape_),
			make_tensmap(then.data_, then.shape_),
			make_tensmap(otherwise.data_, otherwise.shape_)});
}

/// Only applies to 2-d tensors
/// Apply matrix multiplication of a and b
template <typename T>
EigenptrT<T> matmul (teq::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
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

/// Apply convolution of kernel across input
template <typename T>
EigenptrT<T> convolution (teq::Shape& outshape, const OpArg<T>& input, const OpArg<T>& kernel)
{
	assert(nullptr != kernel.coorder_);
	teq::ShapeT dims;
	kernel.coorder_->access(
		[&](const teq::MatrixT& args)
		{
			for (teq::RankT i = 0; i < teq::rank_cap; ++i)
			{
				dims[i] = args[0][i];
			}
		});

	return make_eigentensor<T,Eigen::TensorConvolutionOp<
		const teq::ShapeT,
		const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape),
		[&](std::vector<TensMapT<T>>& args)
		{
			return args[0].convolve(args[1], dims);
		}, {
			make_tensmap(input.data_, input.shape_),
			make_tensmap(kernel.data_, kernel.shape_)});
}

/// Applies the gradient of convolution with respect to image
template <typename T>
EigenptrT<T> convolution_image_grad (teq::Shape& imageshape,
	const OpArg<T>& kernel, const OpArg<T>& super_composite)
{
	return make_eigentensor<T,
		Eigen::TensorReductionOp<Eigen::internal::SumReducer<T>,
			const teq::ShapeT,
			const Eigen::TensorCwiseBinaryOp<
				Eigen::internal::scalar_product_op<T,T>,
				const Eigen::TensorBroadcastingOp<
					const std::array<teq::DimT,teq::rank_cap+1>,
					const Eigen::TensorReshapingOp<
						const std::array<teq::DimT,teq::rank_cap+1>,
						Eigen::TensorReverseOp<
							const std::array<bool,teq::rank_cap>,
							TensMapT<T>
						>
					>
				>,
				const Eigen::TensorPatchOp<
					const teq::ShapeT,
					const Eigen::TensorPaddingOp<
						const std::array<std::pair<int,int>,teq::rank_cap>,
						const TensMapT<T>
					>
				>
			>
		>,
		std::vector<TensMapT<T>>>(shape_convert(imageshape),
		[&](std::vector<TensMapT<T>>& args)
		{
			auto& outshape = super_composite.shape_;

			teq::ShapeT patch_dims;
			std::copy(outshape.begin(), outshape.end(), patch_dims.begin());
			Eigen::array<std::pair<int,int>,teq::rank_cap> paddings;
			for (teq::RankT i = 0; i < teq::rank_cap; ++i)
			{
				int paddsize = outshape.at(i) - 1;
				paddings[i] = std::make_pair(paddsize, paddsize);
			}
			auto patched = args[0].pad(paddings)
				.extract_patches(patch_dims);

			std::array<bool,teq::rank_cap> revflags;
			std::fill(revflags.begin(), revflags.end(), true);
			std::array<teq::DimT,teq::rank_cap+1> pshape;
			std::copy(outshape.begin(), outshape.end(), pshape.begin());
			pshape[teq::rank_cap] = 1;
			std::array<teq::DimT,teq::rank_cap+1> expansion;
			std::fill(expansion.begin(), expansion.end(), 1);
			expansion[teq::rank_cap] = imageshape.n_elems();
			auto partial = args[1]
				.reverse(revflags)
				.reshape(pshape)
				.broadcast(expansion) * patched;

			teq::ShapeT shapespace;
			std::iota(shapespace.begin(), shapespace.end(), 0);
			return partial.sum(shapespace);
		}, {
			make_tensmap(kernel.data_, kernel.shape_),
			make_tensmap(super_composite.data_, super_composite.shape_)});
}

/// Applies the gradient of convolution with respect to kernel
template <typename T>
EigenptrT<T> convolution_kernel_grad (teq::Shape& kernelshape,
	const OpArg<T>& image, const OpArg<T>& super_composite)
{
	return make_eigentensor<T,
		Eigen::TensorReductionOp<Eigen::internal::SumReducer<T>,
			const teq::ShapeT,
			const Eigen::TensorCwiseBinaryOp<
				Eigen::internal::scalar_product_op<T,T>,
				const Eigen::TensorBroadcastingOp<
					const std::array<teq::DimT,teq::rank_cap+1>,
					const Eigen::TensorReshapingOp<
						const std::array<teq::DimT,teq::rank_cap+1>,
						TensMapT<T>
					>
				>,
				const Eigen::TensorPatchOp<
					const teq::ShapeT,
					const TensMapT<T>
				>
			>
		>,
		std::vector<TensMapT<T>>>(shape_convert(kernelshape),
		[&](std::vector<TensMapT<T>>& args)
		{
			auto& outshape = super_composite.shape_;

			teq::ShapeT patch_dims;
			std::copy(outshape.begin(), outshape.end(), patch_dims.begin());
			auto patched = args[0].extract_patches(patch_dims);

			std::array<teq::DimT,teq::rank_cap+1> pshape;
			std::copy(outshape.begin(), outshape.end(), pshape.begin());
			pshape[teq::rank_cap] = 1;
			std::array<teq::DimT,teq::rank_cap+1> expansion;
			std::fill(expansion.begin(), expansion.end(), 1);
			expansion[teq::rank_cap] = kernelshape.n_elems();
			auto partial = args[1]
				.reshape(pshape)
				.broadcast(expansion) * patched;

			teq::ShapeT shapespace;
			std::iota(shapespace.begin(), shapespace.end(), 0);
			return partial.sum(shapespace);
		}, {
			make_tensmap(image.data_, image.shape_),
			make_tensmap(super_composite.data_, super_composite.shape_)});
}

}

#endif // ETEQ_OPERATOR_HPP
