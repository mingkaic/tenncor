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
template <size_t N, typename T=teq::RankT>
inline std::array<T,N> dim_copy (std::vector<T> d)
{
	std::array<T,N> out;
	auto it = d.begin();
	std::copy(it, it + N, out.begin());
	return out;
}

#define _ETEQ_INTERNAL_V2A_CASE(N, PROCESS, RED)\
case N: return make_eigentensor<T,\
Eigen::TensorReshapingOp<const DimensionsT,const ReduceOutT<RED,N,T>>,TensMapT<T>>(\
outdims, make_tensmap(in.data_, in.shape_), [vdims, &outdims](TensMapT<T>& in) {\
return in.PROCESS(::eteq::internal::dim_copy<N>(vdims)).reshape(outdims); });

#define _ETEQ_INTERNAL_V2A(PROCESS, RED) {\
	assert(nullptr != in.coorder_);\
	std::vector<teq::RankT> vdims;\
	vdims.reserve(teq::rank_cap);\
	in.coorder_->access([&](const teq::MatrixT& args){\
		for (teq::RankT i = 0; i < teq::rank_cap &&\
			args[0][i] < teq::rank_cap; ++i)\
		{ vdims.push_back(args[0][i]); }\
	});\
	DimensionsT outdims = shape_convert(outshape);\
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
	} return make_eigentensor<T,Eigen::TensorReshapingOp<\
		const DimensionsT,const ReduceOutT<RED,8,T>>,TensMapT<T>>(\
		outdims, make_tensmap(in.data_, in.shape_),\
		[vdims, &outdims](TensMapT<T>& in) {\
			return in.PROCESS(::eteq::internal::dim_copy<8>(vdims)).reshape(outdims);\
		});\
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

/// Return Eigen data object that argmax in tensor at return_dim
template <typename T>
EigenptrT<T> argmax (teq::Shape& outshape, const OpArg<T>& in)
{
	assert(nullptr != in.coorder_);
	teq::RankT return_dim;
	in.coorder_->access(
		[&](const teq::MatrixT& args)
		{
			return_dim = args[0][0];
		});
	DimensionsT outdims = shape_convert(outshape);
	if (return_dim >= teq::rank_cap)
	{
		return make_eigentensor<T,Eigen::TensorReshapingOp<
			const DimensionsT,
			const Eigen::TensorConversionOp<T,
				const Eigen::TensorTupleReducerOp<
					Eigen::internal::ArgMaxTupleReducer<
						Eigen::Tuple<Eigen::Index,T>>,
					const Eigen::array<Eigen::Index,teq::rank_cap>,
					const TensMapT<T>>>>,
			TensMapT<T>>(
			outdims, make_tensmap(in.data_, in.shape_),
			[&outdims](TensMapT<T>& in)
			{
				return in.argmax().template cast<T>().reshape(outdims);
			});
	}
	return make_eigentensor<T,Eigen::TensorReshapingOp<
		const DimensionsT,
		const Eigen::TensorConversionOp<T,
			const Eigen::TensorTupleReducerOp<
				Eigen::internal::ArgMaxTupleReducer<
					Eigen::Tuple<Eigen::Index,T>>,
				const Eigen::array<Eigen::Index,1>,
				const TensMapT<T>>>>,
		TensMapT<T>>(
		outdims, make_tensmap(in.data_, in.shape_),
		[return_dim, &outdims](TensMapT<T>& in)
		{
			return in.argmax(return_dim).template cast<T>().reshape(outdims);
		});
}

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
		shape_convert(outshape), make_tensmap(in.data_, in.shape_),
		[coord](TensMapT<T>& in)
		{
			return in.broadcast(coord);
		});
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
			shape_convert(outshape), make_matmap(in.data_, in.shape_),
			[](MatMapT<T>& in)
			{
				return in.transpose();
			});
	}
	return make_eigentensor<T,Eigen::TensorShufflingOp<
		const teq::CoordT,TensMapT<T>>,TensMapT<T>>(
		shape_convert(outshape), make_tensmap(in.data_, in.shape_),
		[reorder](TensMapT<T>& in)
		{
			return in.shuffle(reorder);
		});
}

/// Return Eigen data object that reshapes
template <typename T>
EigenptrT<T> reshape (teq::Shape& outshape, const OpArg<T>& in)
{
	return make_eigentensor<T,TensMapT<T>,TensMapT<T>>(
		shape_convert(outshape), make_tensmap(in.data_, in.shape_),
		[](TensMapT<T>& in){ return in; });
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
	DimensionsT outdims = shape_convert(outshape);
	return make_eigentensor<T,Eigen::TensorReshapingOp<
			const DimensionsT,
			Eigen::TensorSlicingOp<
				const teq::ShapeT, const teq::ShapeT,
				TensMapT<T>>>,
		TensMapT<T>>(
		outdims, make_tensmap(in.data_, in.shape_),
		[&offsets, &extents, &outdims](TensMapT<T>& in)
		{
			return in.slice(offsets, extents).reshape(outdims);
		});
}

template <typename T>
EigenptrT<T> group_concat (teq::Shape& outshape, const std::vector<OpArg<T>>& group)
{
	assert(group.size() > 1);
	auto coorder = group[0].coorder_;
	assert(nullptr != coorder);
	teq::RankT dimension;
	coorder->access(
		[&](const teq::MatrixT& args)
		{
			dimension = args[0][0];
		});
	std::vector<TensMapT<T>> args;
	args.reserve(group.size());
	std::transform(group.begin(), group.end(), std::back_inserter(args),
		[](const OpArg<T>& arg)
		{
			return make_tensmap(arg.data_, arg.shape_);
		});
	std::array<Eigen::Index,teq::rank_cap-1> reshaped;
	auto it = outshape.begin();
	std::copy(it, it + dimension, reshaped.begin());
	std::copy(it + dimension + 1, outshape.end(), reshaped.begin() + dimension);
	return std::make_shared<EigenAssignTens<T,std::vector<TensMapT<T>>>>(
		shape_convert(outshape), args,
		[dimension,reshaped](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
		{
			for (size_t i = 0, n = args.size(); i < n; ++i)
			{
				out.chip(i, dimension) = args[i].reshape(reshaped);
			}
		});
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
		shape_convert(outshape), make_tensmap(in.data_, in.shape_),
		[&paddings](TensMapT<T>& in)
		{
			return in.pad(paddings);
		});
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
		shape_convert(outshape), make_tensmap(in.data_, in.shape_),
		[&incrs](TensMapT<T>& in)
		{
			return in.stride(incrs);
		});
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
	return std::make_shared<EigenAssignTens<T,TensMapT<T>>>(shape_convert(outshape),
		make_tensmap(in.data_, in.shape_),
		[incrs](TensorT<T>& out, const TensMapT<T>& in)
		{
			out.stride(incrs) = in;
		});
}

template <typename T>
EigenptrT<T> reverse (teq::Shape& outshape, const OpArg<T>& in)
{
	assert(nullptr != in.coorder_);
	std::array<bool,teq::rank_cap> do_reverse;
	std::fill(do_reverse.begin(), do_reverse.end(), false);
	in.coorder_->access(
		[&](const teq::MatrixT& args)
		{
			for (teq::RankT i = 0; i < teq::rank_cap; ++i)
			{
				if (false == std::isnan(args[0][i]))
				{
					do_reverse[args[0][i]] = true;
				}
			}
		});
	return make_eigentensor<T,Eigen::TensorReverseOp<
			const std::array<bool,teq::rank_cap>,
			TensMapT<T>
		>,
		TensMapT<T>>(
		shape_convert(outshape), make_tensmap(in.data_, in.shape_),
		[&do_reverse](TensMapT<T>& in)
		{
			return in.reverse(do_reverse);
		});
}

template <typename T>
EigenptrT<T> concat (teq::Shape& outshape,
	const OpArg<T>& left, const OpArg<T>& right)
{
	assert(nullptr != left.coorder_);
	teq::RankT axis;
	left.coorder_->access(
		[&](const teq::MatrixT& args)
		{
			axis = args[0][0];
		});
	return make_eigentensor<T,
		Eigen::TensorConcatenationOp<
			const teq::RankT,TensMapT<T>,TensMapT<T>>,
		std::vector<TensMapT<T>>>(
		shape_convert(outshape), {
			make_tensmap(left.data_, left.shape_),
			make_tensmap(right.data_, right.shape_)},
		[axis](std::vector<TensMapT<T>>& args)
		{
			return args[0].concatenate(args[1], axis);
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
			make_matmap(in.data_, in.shape_),
			[](MatMapT<T>& in)
			{
				return in.cwiseAbs();
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_abs_op<T>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		make_tensmap(in.data_, in.shape_),
		[](TensMapT<T>& in)
		{
			return in.abs();
		});
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
			make_matmap(in.data_, in.shape_),
			[](MatMapT<T>& in)
			{
				return -in;
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_opposite_op<T>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		make_tensmap(in.data_, in.shape_),
		[](TensMapT<T>& in)
		{
			return -in;
		});
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
				make_matmap(in.data_, in.shape_),
				[](MatMapT<T>& in)
				{
					return in.array().sin();
				});
		}
	}
#endif
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		std::function<T(const T&)>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		make_tensmap(in.data_, in.shape_),
		[](TensMapT<T>& in)
		{
			return in.unaryExpr(std::function<T(const T&)>(
				[](const T& a) -> T
				{
					return std::sin(a);
				}));
		});
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
				make_matmap(in.data_, in.shape_),
				[](MatMapT<T>& in)
				{
					return in.array().cos();
				});
		}
	}
#endif
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		std::function<T(const T&)>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		make_tensmap(in.data_, in.shape_),
		[](TensMapT<T>& in)
		{
			return in.unaryExpr(std::function<T(const T&)>(
				[](const T& a) -> T
				{
					return std::cos(a);
				}));
		});
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
			make_matmap(in.data_, in.shape_),
			[](MatMapT<T>& in)
			{
				return in.array().tan();
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		std::function<T(const T&)>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		make_tensmap(in.data_, in.shape_),
		[](TensMapT<T>& in)
		{
			return in.unaryExpr(std::function<T(const T&)>(
				[](const T& a) -> T
				{
					return std::tan(a);
				}));
		});
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
			make_matmap(in.data_, in.shape_),
			[](MatMapT<T>& in)
			{
				return in.array().exp();
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_exp_op<T>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		make_tensmap(in.data_, in.shape_),
		[](TensMapT<T>& in)
		{
			return in.exp();
		});
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
			make_matmap(in.data_, in.shape_),
			[](MatMapT<T>& in)
			{
				return in.array().log();
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_log_op<T>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		make_tensmap(in.data_, in.shape_),
		[](TensMapT<T>& in)
		{
			return in.log();
		});
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
			make_matmap(in.data_, in.shape_),
			[](MatMapT<T>& in)
			{
				return in.cwiseSqrt();
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_sqrt_op<T>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		make_tensmap(in.data_, in.shape_),
		[](TensMapT<T>& in)
		{
			return in.sqrt();
		});
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
			make_matmap(in.data_, in.shape_),
			[](MatMapT<T>& in)
			{
				return in.array().round();
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_round_op<T>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		make_tensmap(in.data_, in.shape_),
		[](TensMapT<T>& in)
		{
			return in.round();
		});
}

template <typename T>
EigenptrT<T> sigmoid (teq::Shape& outshape, const OpArg<T>& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseUnaryOp<
			Eigen::internal::scalar_logistic_op<T>,const MatMapT<T>>,
			MatMapT<T>>(shape_convert(outshape),
			make_matmap(in.data_, in.shape_),
			[](MatMapT<T>& in)
			{
				return in.unaryExpr(Eigen::internal::scalar_logistic_op<T>());
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_logistic_op<T>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		make_tensmap(in.data_, in.shape_),
		[](TensMapT<T>& in)
		{
			return in.sigmoid();
		});
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
			make_matmap(in.data_, in.shape_),
			[](MatMapT<T>& in)
			{
				return in.unaryExpr(Eigen::internal::scalar_tanh_op<T>());
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_tanh_op<T>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		make_tensmap(in.data_, in.shape_),
		[](TensMapT<T>& in)
		{
			return in.tanh();
		});
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
			make_matmap(in.data_, in.shape_),
			[](MatMapT<T>& in)
			{
				return in.unaryExpr(Eigen::internal::scalar_square_op<T>());
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_square_op<T>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		make_tensmap(in.data_, in.shape_),
		[](TensMapT<T>& in)
		{
			return in.square();
		});
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
			make_matmap(in.data_, in.shape_),
			[](MatMapT<T>& in)
			{
				return in.unaryExpr(Eigen::internal::scalar_cube_op<T>());
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_cube_op<T>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		make_tensmap(in.data_, in.shape_),
		[](TensMapT<T>& in)
		{
			return in.cube();
		});
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
			std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap(a.data_, a.shape_),
				make_matmap(b.data_, b.shape_)},
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0].binaryExpr(args[1],
					std::function<T(const T&,const T&)>(
					[](const T& a, const T& b) -> T
					{
						return std::pow(a, b);
					}));
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseBinaryOp<
		std::function<T(const T&,const T&)>,
		const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap(a.data_, a.shape_),
			make_tensmap(b.data_, b.shape_)},
		[](std::vector<TensMapT<T>>& args)
		{
			return args[0].binaryExpr(args[1],
				std::function<T(const T&,const T&)>(
				[](const T& a, const T& b) -> T
				{
					return std::pow(a, b);
				}));
		});
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
			std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap(a.data_, a.shape_),
				make_matmap(b.data_, b.shape_)},
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0] + args[1];
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseBinaryOp<
		Eigen::internal::scalar_sum_op<T>,
		const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap(a.data_, a.shape_),
			make_tensmap(b.data_, b.shape_)},
		[](std::vector<TensMapT<T>>& args)
		{
			return args[0] + args[1];
		});
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
			std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap(a.data_, a.shape_),
				make_matmap(b.data_, b.shape_)},
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0] - args[1];
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseBinaryOp<
		Eigen::internal::scalar_difference_op<T>,
		const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap(a.data_, a.shape_),
			make_tensmap(b.data_, b.shape_)},
		[](std::vector<TensMapT<T>>& args)
		{
			return args[0] - args[1];
		});
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
			std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap(a.data_, a.shape_),
				make_matmap(b.data_, b.shape_)},
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0].cwiseProduct(args[1]);
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseBinaryOp<
		Eigen::internal::scalar_product_op<T>,
		const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap(a.data_, a.shape_),
			make_tensmap(b.data_, b.shape_)},
		[](std::vector<TensMapT<T>>& args)
		{
			return args[0] * args[1];
		});
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
			std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap(a.data_, a.shape_),
				make_matmap(b.data_, b.shape_)},
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0].cwiseQuotient(args[1]);
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseBinaryOp<
		Eigen::internal::scalar_quotient_op<T>,
		const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap(a.data_, a.shape_),
			make_tensmap(b.data_, b.shape_)},
		[](std::vector<TensMapT<T>>& args)
		{
			return args[0] / args[1];
		});
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
			std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap(a.data_, a.shape_),
				make_matmap(b.data_, b.shape_)},
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0].binaryExpr(args[1],
					std::function<T(const T&,const T&)>(
					[](const T& a, const T& b) -> T
					{
						return a == b;
					}));
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseBinaryOp<
		std::function<T(const T&,const T&)>,
		const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap(a.data_, a.shape_),
			make_tensmap(b.data_, b.shape_)},
		[](std::vector<TensMapT<T>>& args)
		{
			return args[0].binaryExpr(args[1],
				std::function<T(const T&,const T&)>(
				[](const T& a, const T& b) -> T
				{
					return a == b;
				}));
		});
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
			std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap(a.data_, a.shape_),
				make_matmap(b.data_, b.shape_)},
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0].binaryExpr(args[1],
					std::function<T(const T&,const T&)>(
					[](const T& a, const T& b) -> T
					{
						return a != b;
					}));
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseBinaryOp<
		std::function<T(const T&,const T&)>,
		const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap(a.data_, a.shape_),
			make_tensmap(b.data_, b.shape_)},
		[](std::vector<TensMapT<T>>& args)
		{
			return args[0].binaryExpr(args[1],
				std::function<T(const T&,const T&)>(
				[](const T& a, const T& b) -> T
				{
					return a != b;
				}));
		});
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
			std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap(a.data_, a.shape_),
				make_matmap(b.data_, b.shape_)},
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0].binaryExpr(args[1],
					std::function<T(const T&,const T&)>(
					[](const T& a, const T& b) -> T
					{
						return a < b;
					}));
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseBinaryOp<
		std::function<T(const T&,const T&)>,
		const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap(a.data_, a.shape_),
			make_tensmap(b.data_, b.shape_)},
		[](std::vector<TensMapT<T>>& args)
		{
			return args[0].binaryExpr(args[1],
				std::function<T(const T&,const T&)>(
				[](const T& a, const T& b) -> T
				{
					return a < b;
				}));
		});
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
			std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap(a.data_, a.shape_),
				make_matmap(b.data_, b.shape_)},
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0].binaryExpr(args[1],
					std::function<T(const T&,const T&)>(
					[](const T& a, const T& b) -> T
					{
						return a > b;
					}));
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseBinaryOp<
		std::function<T(const T&,const T&)>,
		const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap(a.data_, a.shape_),
			make_tensmap(b.data_, b.shape_)},
		[](std::vector<TensMapT<T>>& args)
		{
			return args[0].binaryExpr(args[1],
				std::function<T(const T&,const T&)>(
				[](const T& a, const T& b) -> T
				{
					return a > b;
				}));
		});
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
			std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap(a.data_, a.shape_),
				make_matmap(b.data_, b.shape_)},
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0].cwiseMin(args[1]);
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseBinaryOp<
		Eigen::internal::scalar_min_op<T>,
		const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap(a.data_, a.shape_),
			make_tensmap(b.data_, b.shape_)},
		[](std::vector<TensMapT<T>>& args)
		{
			return args[0].cwiseMin(args[1]);
		});
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
			std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap(a.data_, a.shape_),
				make_matmap(b.data_, b.shape_)},
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0].cwiseMax(args[1]);
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseBinaryOp<
		Eigen::internal::scalar_max_op<T>,
		const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap(a.data_, a.shape_),
			make_tensmap(b.data_, b.shape_)},
		[](std::vector<TensMapT<T>>& args)
		{
			return args[0].cwiseMax(args[1]);
		});
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
			std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap(a.data_, a.shape_),
				make_matmap(b.data_, b.shape_)},
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0].binaryExpr(args[1],
					std::function<T(const T&,const T&)>(unif<T>));
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseBinaryOp<
		std::function<T(const T&,const T&)>,
		const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap(a.data_, a.shape_),
			make_tensmap(b.data_, b.shape_)},
		[](std::vector<TensMapT<T>>& args)
		{
			return args[0].binaryExpr(args[1],
				std::function<T(const T&,const T&)>(unif<T>));
		});
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
			std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap(condition.data_, condition.shape_),
				make_matmap(then.data_, then.shape_),
				make_matmap(otherwise.data_, otherwise.shape_)},
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0].select(args[1], args[2]);
			});
	}
	return make_eigentensor<T,
		Eigen::TensorSelectOp<const TensMapT<T>,
			const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap(condition.data_, condition.shape_),
			make_tensmap(then.data_, then.shape_),
			make_tensmap(otherwise.data_, otherwise.shape_)},
		[](std::vector<TensMapT<T>>& args)
		{
			return args[0].select(args[1], args[2]);
		});
}

template <size_t N, typename T>
using ContractionRetT = Eigen::TensorReshapingOp<
	const DimensionsT,
	const Eigen::TensorContractionOp<
		const std::array<std::pair<teq::RankT,teq::RankT>,N>,
		const TensMapT<T>,const TensMapT<T>>>;

/// Only applies to 2-d tensors
/// Apply matrix multiplication of a and b
template <typename T>
EigenptrT<T> matmul (teq::Shape& outshape, const OpArg<T>& a, const OpArg<T>& b)
{
	assert(nullptr != a.coorder_);
	std::vector<std::pair<teq::RankT,teq::RankT>> dims;
	a.coorder_->access(
		[&](const teq::MatrixT& args)
		{
			for (teq::RankT i = 0; i < teq::rank_cap &&
				args[0][i] < teq::rank_cap; ++i)
			{
				// eigen contract apparently reverses left-right relationship
				dims.push_back({args[1][i], args[0][i]});
			}
		});
	if (dims.empty())
	{
		logs::fatal("cannot contract tensors without specified dimensions");
	}
	if (is_2d(a.shape_) && is_2d(b.shape_) &&
		dims.size() == 1 && dims[0].first == 1 && dims[0].second == 0)
	{
		return make_eigenmatrix<T,Eigen::Product<MatMapT<T>,MatMapT<T>>,
			std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap(a.data_, a.shape_),
				make_matmap(b.data_, b.shape_)},
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0] * args[1];
			});
	}
	DimensionsT outdims = shape_convert(outshape);
	std::vector<TensMapT<T>> args = {
		make_tensmap(a.data_, a.shape_),
		make_tensmap(b.data_, b.shape_)};
	switch (dims.size())
	{
		case 1:
			return make_eigentensor<T,ContractionRetT<1,T>,std::vector<TensMapT<T>>>(outdims, args,
				[&](std::vector<TensMapT<T>>& args)
				{
					return args[1].contract(args[0], internal::dim_copy<1>(dims)).reshape(outdims);
				});
		case 2:
			return make_eigentensor<T,ContractionRetT<2,T>,std::vector<TensMapT<T>>>(outdims, args,
				[&](std::vector<TensMapT<T>>& args)
				{
					return args[1].contract(args[0], internal::dim_copy<2>(dims)).reshape(outdims);
				});
		case 3:
			return make_eigentensor<T,ContractionRetT<3,T>,std::vector<TensMapT<T>>>(outdims, args,
				[&](std::vector<TensMapT<T>>& args)
				{
					return args[1].contract(args[0], internal::dim_copy<3>(dims)).reshape(outdims);
				});
		case 4:
			return make_eigentensor<T,ContractionRetT<4,T>,std::vector<TensMapT<T>>>(outdims, args,
				[&](std::vector<TensMapT<T>>& args)
				{
					return args[1].contract(args[0], internal::dim_copy<4>(dims)).reshape(outdims);
				});
		case 5:
			return make_eigentensor<T,ContractionRetT<5,T>,std::vector<TensMapT<T>>>(outdims, args,
				[&](std::vector<TensMapT<T>>& args)
				{
					return args[1].contract(args[0], internal::dim_copy<5>(dims)).reshape(outdims);
				});
		case 6:
			return make_eigentensor<T,ContractionRetT<6,T>,std::vector<TensMapT<T>>>(outdims, args,
				[&](std::vector<TensMapT<T>>& args)
				{
					return args[1].contract(args[0], internal::dim_copy<6>(dims)).reshape(outdims);
				});
		case 7:
			return make_eigentensor<T,ContractionRetT<7,T>,std::vector<TensMapT<T>>>(outdims, args,
				[&](std::vector<TensMapT<T>>& args)
				{
					return args[1].contract(args[0], internal::dim_copy<7>(dims)).reshape(outdims);
				});
		default:
			break;
	}
	return make_eigentensor<T,ContractionRetT<teq::rank_cap,T>,std::vector<TensMapT<T>>>(outdims, args,
		[&](std::vector<TensMapT<T>>& args)
		{
			return args[1].contract(args[0], internal::dim_copy<teq::rank_cap>(dims)).reshape(outdims);
		});
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
		std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap(input.data_, input.shape_),
			make_tensmap(kernel.data_, kernel.shape_)},
		[&](std::vector<TensMapT<T>>& args)
		{
			return args[0].convolve(args[1], dims);
		});
}

}

#endif // ETEQ_OPERATOR_HPP
