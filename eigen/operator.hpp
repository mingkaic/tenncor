///
/// operator.hpp
/// eigen
///
/// Purpose:
/// Define functions manipulating tensor data values
/// No function in this file makes any attempt to check for nullptrs
///
// todo: make this generated

#include "eigen/eigen.hpp"
#include "eigen/random.hpp"
#include "eigen/edge.hpp"

#ifndef EIGEN_OPERATOR_HPP
#define EIGEN_OPERATOR_HPP

namespace eigen
{

template <typename T>
using PairVecT = std::vector<std::pair<T,T>>;

template <typename T>
std::string to_string (const PairVecT<T>& pairs)
{
	PairVecT<int> readable_pairs(pairs.begin(), pairs.end());
	return fmts::to_string(readable_pairs.begin(), readable_pairs.end());
}

template <typename T>
std::vector<double> encode_pair (const PairVecT<T>& pairs)
{
	size_t npairs = pairs.size();
	std::vector<double> out;
	out.reserve(npairs * 2);
	for (auto& p : pairs)
	{
		out.push_back(p.first);
		out.push_back(p.second);
	}
	return out;
}

template <typename T>
const PairVecT<T> decode_pair (std::vector<double>& encoding)
{
	PairVecT<T> out;
	size_t n = encoding.size();
	if (1 == n % 2)
	{
		logs::fatalf("cannot decode odd vector %s into vec of pairs",
			fmts::to_string(encoding.begin(), encoding.end()).c_str());
	}
	out.reserve(n / 2);
	for (size_t i = 0; i < n; i += 2)
	{
		out.push_back({encoding[i], encoding[i + 1]});
	}
	return out;
}

static inline bool is_2d (teq::Shape shape)
{
	return std::all_of(shape.begin() + 2, shape.end(),
		[](teq::DimT dim) { return 1 == dim; });
}

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

}

#define _ARRAY_SWITCH(ARR, CASE)switch (ARR.size()) {\
	case 0: logs::fatal("missing dimensions");\
	case 1: CASE(1)\
	case 2: CASE(2)\
	case 3: CASE(3)\
	case 4: CASE(4)\
	case 5: CASE(5)\
	case 6: CASE(6)\
	case 7: CASE(7)\
	default: break;\
} CASE(8)

#define _EIGEN_RSUM_CASE(N)\
return make_eigentensor<T,\
Eigen::TensorReshapingOp<const DimensionsT,const ReduceOutT<\
Eigen::internal::SumReducer<T>,N,T>>,TensMapT<T>>(\
outdims, make_tensmap(in.data(), in.shape()), [vdims, &outdims](TensMapT<T>& in) {\
return in.sum(::eigen::internal::dim_copy<N>(vdims)).reshape(outdims); });

/// Return Eigen data object representing reduction where aggregation is sum
template <typename T>
EigenptrT<T> reduce_sum (teq::Shape outshape, const iEigenEdge<T>& in, const marsh::Maps& attrs)
{
	auto cdims = get_coorder(attrs);
	std::vector<teq::RankT> vdims(cdims.begin(), cdims.end());
	DimensionsT outdims = shape_convert(outshape);
	_ARRAY_SWITCH(vdims, _EIGEN_RSUM_CASE)
}

#undef _EIGEN_RSUM_CASE

#define _EIGEN_RPROD_CASE(N)\
return make_eigentensor<T,\
Eigen::TensorReshapingOp<const DimensionsT,const ReduceOutT<\
Eigen::internal::ProdReducer<T>,N,T>>,TensMapT<T>>(\
outdims, make_tensmap(in.data(), in.shape()), [vdims, &outdims](TensMapT<T>& in) {\
return in.prod(::eigen::internal::dim_copy<N>(vdims)).reshape(outdims); });

/// Return Eigen data object representing reduction where aggregation is prod
template <typename T>
EigenptrT<T> reduce_prod (teq::Shape outshape, const iEigenEdge<T>& in, const marsh::Maps& attrs)
{
	auto cdims = get_coorder(attrs);
	std::vector<teq::RankT> vdims(cdims.begin(), cdims.end());
	DimensionsT outdims = shape_convert(outshape);
	_ARRAY_SWITCH(vdims, _EIGEN_RPROD_CASE)
}

#undef _EIGEN_RPROD_CASE

#define _EIGEN_RMIN_CASE(N)\
return make_eigentensor<T,\
Eigen::TensorReshapingOp<const DimensionsT,const ReduceOutT<\
Eigen::internal::MinReducer<T>,N,T>>,TensMapT<T>>(\
outdims, make_tensmap(in.data(), in.shape()), [vdims, &outdims](TensMapT<T>& in) {\
return in.minimum(::eigen::internal::dim_copy<N>(vdims)).reshape(outdims); });

/// Return Eigen data object representing reduction where aggregation is min
template <typename T>
EigenptrT<T> reduce_min (teq::Shape outshape, const iEigenEdge<T>& in, const marsh::Maps& attrs)
{
	auto cdims = get_coorder(attrs);
	std::vector<teq::RankT> vdims(cdims.begin(), cdims.end());
	DimensionsT outdims = shape_convert(outshape);
	_ARRAY_SWITCH(vdims, _EIGEN_RMIN_CASE)
}

#undef _EIGEN_RMIN_CASE

#define _EIGEN_RMAX_CASE(N)\
return make_eigentensor<T,\
Eigen::TensorReshapingOp<const DimensionsT,const ReduceOutT<\
Eigen::internal::MaxReducer<T>,N,T>>,TensMapT<T>>(\
outdims, make_tensmap(in.data(), in.shape()), [vdims, &outdims](TensMapT<T>& in) {\
return in.maximum(::eigen::internal::dim_copy<N>(vdims)).reshape(outdims); });

/// Return Eigen data object representing reduction where aggregation is max
template <typename T>
EigenptrT<T> reduce_max (teq::Shape outshape, const iEigenEdge<T>& in, const marsh::Maps& attrs)
{
	auto cdims = get_coorder(attrs);
	std::vector<teq::RankT> vdims(cdims.begin(), cdims.end());
	DimensionsT outdims = shape_convert(outshape);
	_ARRAY_SWITCH(vdims, _EIGEN_RMAX_CASE)
}

#undef _EIGEN_RMIN_CASE

/// Return Eigen data object that argmax in tensor at return_dim
template <typename T>
EigenptrT<T> argmax (teq::Shape outshape, const iEigenEdge<T>& in, const marsh::Maps& attrs)
{
	teq::RankT return_dim = get_coorder(attrs)[0];
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
			outdims, make_tensmap(in.data(), in.shape()),
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
		outdims, make_tensmap(in.data(), in.shape()),
		[return_dim, &outdims](TensMapT<T>& in)
		{
			return in.argmax(return_dim).template cast<T>().reshape(outdims);
		});
}

/// Return Eigen data object representing data broadcast across dimensions
template <typename T>
EigenptrT<T> extend (teq::Shape outshape, const iEigenEdge<T>& in, const marsh::Maps& attrs)
{
	teq::CoordT coord;
	auto c = get_coorder(attrs);
	std::fill(coord.begin(), coord.end(), 1);
	std::copy(c.begin(), c.begin() +
		std::min((size_t) teq::rank_cap, c.size()), coord.begin());
	return make_eigentensor<T,Eigen::TensorBroadcastingOp<
		const teq::CoordT,const TensMapT<T>>,TensMapT<T>>(
		shape_convert(outshape), make_tensmap(in.data(), in.shape()),
		[coord](TensMapT<T>& in)
		{
			return in.broadcast(coord);
		});
}

/// Return Eigen data object representing transpose and permutation
template <typename T>
EigenptrT<T> permute (teq::Shape outshape, const iEigenEdge<T>& in, const marsh::Maps& attrs)
{
	teq::CoordT reorder;
	auto c = get_coorder(attrs);
	assert(c.size() == teq::rank_cap);
	std::copy(c.begin(), c.end(), reorder.begin());
	if (is_2d(outshape) && reorder[0] == 1 && reorder[1] == 0)
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::Transpose<MatMapT<T>>,
			MatMapT<T>>(
			shape_convert(outshape), make_matmap(in.data(), in.shape()),
			[](MatMapT<T>& in)
			{
				return in.transpose();
			});
	}
	return make_eigentensor<T,Eigen::TensorShufflingOp<
		const teq::CoordT,TensMapT<T>>,TensMapT<T>>(
		shape_convert(outshape), make_tensmap(in.data(), in.shape()),
		[reorder](TensMapT<T>& in)
		{
			return in.shuffle(reorder);
		});
}

/// Return Eigen data object that reshapes
template <typename T>
EigenptrT<T> reshape (teq::Shape outshape, const iEigenEdge<T>& in)
{
	return make_eigentensor<T,TensMapT<T>,TensMapT<T>>(
		shape_convert(outshape), make_tensmap(in.data(), in.shape()),
		[](TensMapT<T>& in){ return in; });
}

/// Return Eigen data object representing data slicing of dimensions
template <typename T>
EigenptrT<T> slice (teq::Shape outshape, const iEigenEdge<T>& in, const marsh::Maps& attrs)
{
	teq::Shape shape = in.shape();
	auto c = get_coorder(attrs);
	auto encoding = decode_pair<teq::DimT>(c);
	teq::ShapeT offsets;
	teq::ShapeT extents;
	std::fill(offsets.begin(), offsets.end(), 0);
	std::copy(shape.begin(), shape.end(), extents.begin());
	size_t n = std::min(encoding.size(), (size_t) teq::rank_cap);
	for (size_t i = 0; i < n; ++i)
	{
		offsets[i] = encoding[i].first;
		extents[i] = encoding[i].second;
	}
	auto slist = teq::narrow_shape(shape);
	if (slist.size() > 0 && outshape.compatible_before(
		shape, slist.size() - 1))
	{
		teq::RankT lastdim = slist.size() - 1;
		// only slicing the last dimension
		teq::DimT index = offsets[lastdim];
		teq::NElemT batchsize = shape.n_elems() / shape.at(lastdim);
		// SINCE tensor is column major, index of last dimension denote
		// the number of batches before start of output slice
		// (a batch defined as the subtensor of shape shape[:lastdim])
		return std::make_shared<eigen::EigenRef<T>>(
			in.data() + index * batchsize);
	}
	DimensionsT outdims = shape_convert(outshape);
	return make_eigentensor<T,Eigen::TensorReshapingOp<
		const DimensionsT, Eigen::TensorSlicingOp<
			const teq::ShapeT, const teq::ShapeT,
			TensMapT<T>>>,
		TensMapT<T>>(
		outdims, make_tensmap(in.data(), shape),
		[&offsets, &extents, &outdims](TensMapT<T>& in)
		{
			return in.slice(offsets, extents).reshape(outdims);
		});
}

template <typename T>
EigenptrT<T> group_concat (teq::Shape outshape, const EigenEdgesT<T>& group, const marsh::Maps& attrs)
{
	assert(group.size() > 1);
	teq::RankT dimension = get_coorder(attrs)[0];
	std::vector<TensMapT<T>> args;
	args.reserve(group.size());
	std::transform(group.begin(), group.end(), std::back_inserter(args),
		[](const iEigenEdge<T>& arg)
		{
			return make_tensmap(arg.data(), arg.shape());
		});
	std::array<Eigen::Index,teq::rank_cap-1> reshaped;
	auto it = outshape.begin();
	std::copy(it, it + dimension, reshaped.begin());
	std::copy(it + dimension + 1, outshape.end(), reshaped.begin() + dimension);
	return std::make_shared<EigenAssignTens<T,std::vector<TensMapT<T>>>>(
		0, shape_convert(outshape), args,
		[dimension,reshaped](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
		{
			for (size_t i = 0, n = args.size(); i < n; ++i)
			{
				out.chip(i, dimension) = args[i].reshape(reshaped);
			}
		});
}

template <typename T>
EigenptrT<T> group_sum (teq::Shape outshape, const EigenEdgesT<T>& group)
{
	assert(group.size() > 2);
	std::vector<TensMapT<T>> args;
	args.reserve(group.size());
	std::transform(group.begin(), group.end(), std::back_inserter(args),
		[](const iEigenEdge<T>& arg)
		{
			return make_tensmap(arg.data(), arg.shape());
		});
	return std::make_shared<EigenAssignTens<T,std::vector<TensMapT<T>>>>(
		0, shape_convert(outshape), args,
		[](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
		{
			for (size_t i = 0, n = args.size(); i < n; ++i)
			{
				out += args[i];
			}
		});
}

template <typename T>
EigenptrT<T> group_prod (teq::Shape outshape, const EigenEdgesT<T>& group)
{
	assert(group.size() > 2);
	std::vector<TensMapT<T>> args;
	args.reserve(group.size());
	std::transform(group.begin(), group.end(), std::back_inserter(args),
		[](const iEigenEdge<T>& arg)
		{
			return make_tensmap(arg.data(), arg.shape());
		});
	return std::make_shared<EigenAssignTens<T,std::vector<TensMapT<T>>>>(
		1, shape_convert(outshape), args,
		[](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
		{
			for (size_t i = 0, n = args.size(); i < n; ++i)
			{
				out *= args[i];
			}
		});
}

/// Return Eigen data object representing data zero padding
template <typename T>
EigenptrT<T> pad (teq::Shape outshape, const iEigenEdge<T>& in, const marsh::Maps& attrs)
{
	std::array<std::pair<teq::DimT,teq::DimT>,teq::rank_cap> paddings;
	std::fill(paddings.begin(), paddings.end(),
		std::pair<teq::DimT,teq::DimT>{0, 0});
	auto c = get_coorder(attrs);
	auto encoding = decode_pair<teq::DimT>(c);
	std::copy(encoding.begin(), encoding.begin() +
		std::min((size_t) teq::rank_cap, encoding.size()),
		paddings.begin());
	return make_eigentensor<T,Eigen::TensorPaddingOp<
			const std::array<std::pair<teq::DimT,teq::DimT>,teq::rank_cap>,
			const TensMapT<T>
		>,
		TensMapT<T>>(
		shape_convert(outshape), make_tensmap(in.data(), in.shape()),
		[&paddings](TensMapT<T>& in)
		{
			return in.pad(paddings);
		});
}

/// Return Eigen data object representing strided view of in
template <typename T>
EigenptrT<T> stride (teq::Shape outshape, const iEigenEdge<T>& in, const marsh::Maps& attrs)
{
	Eigen::array<Eigen::DenseIndex,teq::rank_cap> incrs;
	std::fill(incrs.begin(), incrs.end(), 1);
	auto c = get_coorder(attrs);
	std::copy(c.begin(), c.begin() + std::min((size_t) teq::rank_cap, c.size()),
		incrs.begin());
	return make_eigentensor<T,Eigen::TensorStridingOp<
			const Eigen::array<Eigen::DenseIndex,teq::rank_cap>,
			TensMapT<T>
		>,
		TensMapT<T>>(
		shape_convert(outshape), make_tensmap(in.data(), in.shape()),
		[&incrs](TensMapT<T>& in)
		{
			return in.stride(incrs);
		});
}

/// Return Eigen data object that scatters data in
/// specific increments across dimensions
/// This function is the reverse of stride
template <typename T>
EigenptrT<T> scatter (teq::Shape outshape, const iEigenEdge<T>& in, const marsh::Maps& attrs)
{
	Eigen::array<Eigen::DenseIndex,teq::rank_cap> incrs;
	std::fill(incrs.begin(), incrs.end(), 1);
	auto c = get_coorder(attrs);
	std::copy(c.begin(), c.begin() + std::min((size_t) teq::rank_cap, c.size()),
		incrs.begin());
	return std::make_shared<EigenAssignTens<T,TensMapT<T>>>(
		0, shape_convert(outshape), make_tensmap(in.data(), in.shape()),
		[incrs](TensorT<T>& out, const TensMapT<T>& in)
		{
			out.stride(incrs) = in;
		});
}

template <typename T>
EigenptrT<T> reverse (teq::Shape outshape, const iEigenEdge<T>& in, const marsh::Maps& attrs)
{
	std::array<bool,teq::rank_cap> do_reverse;
	std::fill(do_reverse.begin(), do_reverse.end(), false);
	auto c = get_coorder(attrs);
	for (teq::RankT i : c)
	{
		do_reverse[i] = true;
	}
	return make_eigentensor<T,Eigen::TensorReverseOp<
			const std::array<bool,teq::rank_cap>,
			TensMapT<T>
		>,
		TensMapT<T>>(
		shape_convert(outshape), make_tensmap(in.data(), in.shape()),
		[&do_reverse](TensMapT<T>& in)
		{
			return in.reverse(do_reverse);
		});
}

template <typename T>
EigenptrT<T> concat (teq::Shape outshape, const iEigenEdge<T>& left, const iEigenEdge<T>& right, const marsh::Maps& attrs)
{
	teq::RankT axis = get_coorder(attrs)[0];
	return make_eigentensor<T,
		Eigen::TensorConcatenationOp<
			const teq::RankT,TensMapT<T>,TensMapT<T>>,
		std::vector<TensMapT<T>>>(
		shape_convert(outshape), {
			make_tensmap(left.data(), left.shape()),
			make_tensmap(right.data(), right.shape())},
		[axis](std::vector<TensMapT<T>>& args)
		{
			return args[0].concatenate(args[1], axis);
		});
}

/// Given reference to output array, and input vector ref,
/// make output elements take absolute value of inputs
template <typename T>
EigenptrT<T> abs (teq::Shape outshape, const iEigenEdge<T>& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseUnaryOp<
			Eigen::internal::scalar_abs_op<T>,const MatMapT<T>>,
			MatMapT<T>>(shape_convert(outshape),
			make_matmap(in.data(), in.shape()),
			[](MatMapT<T>& in)
			{
				return in.cwiseAbs();
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_abs_op<T>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		make_tensmap(in.data(), in.shape()),
		[](TensMapT<T>& in)
		{
			return in.abs();
		});
}

/// Given reference to output array, and input vector ref,
/// make output elements take negatives of inputs
template <typename T>
EigenptrT<T> neg (teq::Shape outshape, const iEigenEdge<T>& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseUnaryOp<
			Eigen::internal::scalar_opposite_op<T>,const MatMapT<T>>,
			MatMapT<T>>(shape_convert(outshape),
			make_matmap(in.data(), in.shape()),
			[](MatMapT<T>& in)
			{
				return -in;
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_opposite_op<T>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		make_tensmap(in.data(), in.shape()),
		[](TensMapT<T>& in)
		{
			return -in;
		});
}

/// Given reference to output array, and input vector ref,
/// make output elements take sine of inputs
template <typename T>
EigenptrT<T> sin (teq::Shape outshape, const iEigenEdge<T>& in)
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
				make_matmap(in.data(), in.shape()),
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
		make_tensmap(in.data(), in.shape()),
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
EigenptrT<T> cos (teq::Shape outshape, const iEigenEdge<T>& in)
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
				make_matmap(in.data(), in.shape()),
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
		make_tensmap(in.data(), in.shape()),
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
EigenptrT<T> tan (teq::Shape outshape, const iEigenEdge<T>& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,
			typename Eigen::ArrayWrapper<MatMapT<T>>::TanReturnType,
			MatMapT<T>>(shape_convert(outshape),
			make_matmap(in.data(), in.shape()),
			[](MatMapT<T>& in)
			{
				return in.array().tan();
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		std::function<T(const T&)>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		make_tensmap(in.data(), in.shape()),
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
EigenptrT<T> exp (teq::Shape outshape, const iEigenEdge<T>& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,
			typename Eigen::ArrayWrapper<MatMapT<T>>::ExpReturnType,
			MatMapT<T>>(shape_convert(outshape),
			make_matmap(in.data(), in.shape()),
			[](MatMapT<T>& in)
			{
				return in.array().exp();
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_exp_op<T>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		make_tensmap(in.data(), in.shape()),
		[](TensMapT<T>& in)
		{
			return in.exp();
		});
}

/// Given reference to output array, and input vector ref,
/// make output elements take natural log of inputs
template <typename T>
EigenptrT<T> log (teq::Shape outshape, const iEigenEdge<T>& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,
			typename Eigen::ArrayWrapper<MatMapT<T>>::LogReturnType,
			MatMapT<T>>(shape_convert(outshape),
			make_matmap(in.data(), in.shape()),
			[](MatMapT<T>& in)
			{
				return in.array().log();
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_log_op<T>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		make_tensmap(in.data(), in.shape()),
		[](TensMapT<T>& in)
		{
			return in.log();
		});
}

/// Given reference to output array, and input vector ref,
/// make output elements take square root of inputs
template <typename T>
EigenptrT<T> sqrt (teq::Shape outshape, const iEigenEdge<T>& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseUnaryOp<
			Eigen::internal::scalar_sqrt_op<T>,const MatMapT<T>>,
			MatMapT<T>>(shape_convert(outshape),
			make_matmap(in.data(), in.shape()),
			[](MatMapT<T>& in)
			{
				return in.cwiseSqrt();
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_sqrt_op<T>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		make_tensmap(in.data(), in.shape()),
		[](TensMapT<T>& in)
		{
			return in.sqrt();
		});
}

/// Given reference to output array, and input vector ref,
/// make output elements take rounded values of inputs
template <typename T>
EigenptrT<T> round (teq::Shape outshape, const iEigenEdge<T>& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,
			typename Eigen::ArrayWrapper<MatMapT<T>>::RoundReturnType,
			MatMapT<T>>(shape_convert(outshape),
			make_matmap(in.data(), in.shape()),
			[](MatMapT<T>& in)
			{
				return in.array().round();
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_round_op<T>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		make_tensmap(in.data(), in.shape()),
		[](TensMapT<T>& in)
		{
			return in.round();
		});
}

template <typename T>
EigenptrT<T> sigmoid (teq::Shape outshape, const iEigenEdge<T>& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseUnaryOp<
			Eigen::internal::scalar_logistic_op<T>,const MatMapT<T>>,
			MatMapT<T>>(shape_convert(outshape),
			make_matmap(in.data(), in.shape()),
			[](MatMapT<T>& in)
			{
				return in.unaryExpr(Eigen::internal::scalar_logistic_op<T>());
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_logistic_op<T>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		make_tensmap(in.data(), in.shape()),
		[](TensMapT<T>& in)
		{
			return in.sigmoid();
		});
}

template <typename T>
EigenptrT<T> tanh (teq::Shape outshape, const iEigenEdge<T>& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseUnaryOp<
			Eigen::internal::scalar_tanh_op<T>,const MatMapT<T>>,
			MatMapT<T>>(shape_convert(outshape),
			make_matmap(in.data(), in.shape()),
			[](MatMapT<T>& in)
			{
				return in.unaryExpr(Eigen::internal::scalar_tanh_op<T>());
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_tanh_op<T>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		make_tensmap(in.data(), in.shape()),
		[](TensMapT<T>& in)
		{
			return in.tanh();
		});
}

template <typename T>
EigenptrT<T> square (teq::Shape outshape, const iEigenEdge<T>& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseUnaryOp<
			Eigen::internal::scalar_square_op<T>,const MatMapT<T>>,
			MatMapT<T>>(shape_convert(outshape),
			make_matmap(in.data(), in.shape()),
			[](MatMapT<T>& in)
			{
				return in.unaryExpr(Eigen::internal::scalar_square_op<T>());
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_square_op<T>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		make_tensmap(in.data(), in.shape()),
		[](TensMapT<T>& in)
		{
			return in.square();
		});
}

template <typename T>
EigenptrT<T> cube (teq::Shape outshape, const iEigenEdge<T>& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseUnaryOp<
			Eigen::internal::scalar_cube_op<T>,const MatMapT<T>>,
			MatMapT<T>>(shape_convert(outshape),
			make_matmap(in.data(), in.shape()),
			[](MatMapT<T>& in)
			{
				return in.unaryExpr(Eigen::internal::scalar_cube_op<T>());
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_cube_op<T>,const TensMapT<T>>,
		TensMapT<T>>(shape_convert(outshape),
		make_tensmap(in.data(), in.shape()),
		[](TensMapT<T>& in)
		{
			return in.cube();
		});
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply std::pow operator
/// Only accept 2 arguments
template <typename T>
EigenptrT<T> pow (teq::Shape outshape, const iEigenEdge<T>& a, const iEigenEdge<T>& b)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseBinaryOp<
			std::function<T(const T&,const T&)>,
			const MatMapT<T>,const MatMapT<T>>,
			std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap(a.data(), a.shape()),
				make_matmap(b.data(), b.shape())},
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
			make_tensmap(a.data(), a.shape()),
			make_tensmap(b.data(), b.shape())},
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
EigenptrT<T> add (teq::Shape outshape, const iEigenEdge<T>& a, const iEigenEdge<T>& b)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseBinaryOp<
			Eigen::internal::scalar_sum_op<T>,
			const MatMapT<T>,const MatMapT<T>>,
			std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap(a.data(), a.shape()),
				make_matmap(b.data(), b.shape())},
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0] + args[1];
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseBinaryOp<
		Eigen::internal::scalar_sum_op<T>,
		const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap(a.data(), a.shape()),
			make_tensmap(b.data(), b.shape())},
		[](std::vector<TensMapT<T>>& args)
		{
			return args[0] + args[1];
		});
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index subtract
/// Only accept 2 arguments
template <typename T>
EigenptrT<T> sub (teq::Shape outshape, const iEigenEdge<T>& a, const iEigenEdge<T>& b)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseBinaryOp<
			Eigen::internal::scalar_difference_op<T>,
			const MatMapT<T>,const MatMapT<T>>,
			std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap(a.data(), a.shape()),
				make_matmap(b.data(), b.shape())},
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0] - args[1];
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseBinaryOp<
		Eigen::internal::scalar_difference_op<T>,
		const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap(a.data(), a.shape()),
			make_tensmap(b.data(), b.shape())},
		[](std::vector<TensMapT<T>>& args)
		{
			return args[0] - args[1];
		});
}

template <typename T>
EigenptrT<T> mul (teq::Shape outshape, const iEigenEdge<T>& a, const iEigenEdge<T>& b)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseBinaryOp<
			Eigen::internal::scalar_product_op<T>,
			const MatMapT<T>,const MatMapT<T>>,
			std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap(a.data(), a.shape()),
				make_matmap(b.data(), b.shape())},
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0].cwiseProduct(args[1]);
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseBinaryOp<
		Eigen::internal::scalar_product_op<T>,
		const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap(a.data(), a.shape()),
			make_tensmap(b.data(), b.shape())},
		[](std::vector<TensMapT<T>>& args)
		{
			return args[0] * args[1];
		});
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index divide
/// Only accept 2 arguments
template <typename T>
EigenptrT<T> div (teq::Shape outshape, const iEigenEdge<T>& a, const iEigenEdge<T>& b)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseBinaryOp<
			Eigen::internal::scalar_quotient_op<T>,
			const MatMapT<T>,const MatMapT<T>>,
			std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap(a.data(), a.shape()),
				make_matmap(b.data(), b.shape())},
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0].cwiseQuotient(args[1]);
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseBinaryOp<
		Eigen::internal::scalar_quotient_op<T>,
		const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap(a.data(), a.shape()),
			make_tensmap(b.data(), b.shape())},
		[](std::vector<TensMapT<T>>& args)
		{
			return args[0] / args[1];
		});
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply == operator
/// Only accept 2 arguments
template <typename T>
EigenptrT<T> eq (teq::Shape outshape, const iEigenEdge<T>& a, const iEigenEdge<T>& b)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseBinaryOp<
			std::function<T(const T&,const T&)>,
			const MatMapT<T>,const MatMapT<T>>,
			std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap(a.data(), a.shape()),
				make_matmap(b.data(), b.shape())},
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
			make_tensmap(a.data(), a.shape()),
			make_tensmap(b.data(), b.shape())},
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
EigenptrT<T> neq (teq::Shape outshape, const iEigenEdge<T>& a, const iEigenEdge<T>& b)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseBinaryOp<
			std::function<T(const T&,const T&)>,
			const MatMapT<T>,const MatMapT<T>>,
			std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap(a.data(), a.shape()),
				make_matmap(b.data(), b.shape())},
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
			make_tensmap(a.data(), a.shape()),
			make_tensmap(b.data(), b.shape())},
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
EigenptrT<T> lt (teq::Shape outshape, const iEigenEdge<T>& a, const iEigenEdge<T>& b)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseBinaryOp<
			std::function<T(const T&,const T&)>,
			const MatMapT<T>,const MatMapT<T>>,
			std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap(a.data(), a.shape()),
				make_matmap(b.data(), b.shape())},
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
			make_tensmap(a.data(), a.shape()),
			make_tensmap(b.data(), b.shape())},
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
EigenptrT<T> gt (teq::Shape outshape, const iEigenEdge<T>& a, const iEigenEdge<T>& b)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseBinaryOp<
			std::function<T(const T&,const T&)>,
			const MatMapT<T>,const MatMapT<T>>,
			std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap(a.data(), a.shape()),
				make_matmap(b.data(), b.shape())},
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
			make_tensmap(a.data(), a.shape()),
			make_tensmap(b.data(), b.shape())},
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
EigenptrT<T> min (teq::Shape outshape, const iEigenEdge<T>& a, const iEigenEdge<T>& b)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseBinaryOp<
			Eigen::internal::scalar_min_op<T,T>,
			const MatMapT<T>,const MatMapT<T>>,
			std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap(a.data(), a.shape()),
				make_matmap(b.data(), b.shape())},
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0].cwiseMin(args[1]);
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseBinaryOp<
		Eigen::internal::scalar_min_op<T>,
		const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap(a.data(), a.shape()),
			make_tensmap(b.data(), b.shape())},
		[](std::vector<TensMapT<T>>& args)
		{
			return args[0].cwiseMin(args[1]);
		});
}

/// Given arguments, for every mapped index i in range [0:max_nelems],
/// take the maximum all elements for all arguments
template <typename T>
EigenptrT<T> max (teq::Shape outshape, const iEigenEdge<T>& a, const iEigenEdge<T>& b)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseBinaryOp<
			Eigen::internal::scalar_max_op<T,T>,
			const MatMapT<T>,const MatMapT<T>>,
			std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap(a.data(), a.shape()),
				make_matmap(b.data(), b.shape())},
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0].cwiseMax(args[1]);
			});
	}
	return make_eigentensor<T,Eigen::TensorCwiseBinaryOp<
		Eigen::internal::scalar_max_op<T>,
		const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap(a.data(), a.shape()),
			make_tensmap(b.data(), b.shape())},
		[](std::vector<TensMapT<T>>& args)
		{
			return args[0].cwiseMax(args[1]);
		});
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply std::uniform_distributon function
/// Only accept 2 arguments
template <typename T>
EigenptrT<T> rand_uniform (teq::Shape outshape, const iEigenEdge<T>& a, const iEigenEdge<T>& b)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::CwiseBinaryOp<
			std::function<T(const T&,const T&)>,
			const MatMapT<T>,const MatMapT<T>>,
			std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap(a.data(), a.shape()),
				make_matmap(b.data(), b.shape())},
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
			make_tensmap(a.data(), a.shape()),
			make_tensmap(b.data(), b.shape())},
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
EigenptrT<T> select (teq::Shape outshape, const iEigenEdge<T>& condition,
	const iEigenEdge<T>& then, const iEigenEdge<T>& otherwise)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return make_eigenmatrix<T,Eigen::Select<MatMapT<T>,
			MatMapT<T>,MatMapT<T>>,std::vector<MatMapT<T>>>(
			shape_convert(outshape), {
			make_matmap(condition.data(), condition.shape()),
			make_matmap(then.data(), then.shape()),
			make_matmap(otherwise.data(), otherwise.shape())},
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0].select(args[1], args[2]);
			});
	}
	return make_eigentensor<T,
		Eigen::TensorSelectOp<const TensMapT<T>,
			const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap(condition.data(), condition.shape()),
			make_tensmap(then.data(), then.shape()),
			make_tensmap(otherwise.data(), otherwise.shape())},
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

#define _EIGEN_MATMUL_CASE(N)\
return make_eigentensor<T,ContractionRetT<N,T>,std::vector<TensMapT<T>>>(outdims, args,\
[&](std::vector<TensMapT<T>>& args){\
return args[1].contract(args[0], internal::dim_copy<N>(dims)).reshape(outdims); });

/// Only applies to 2-d tensors
/// Apply matrix multiplication of a and b
template <typename T>
EigenptrT<T> matmul (teq::Shape outshape, const iEigenEdge<T>& a, const iEigenEdge<T>& b, const marsh::Maps& attrs)
{
	auto c = get_coorder(attrs);
	auto dims = decode_pair<teq::RankT>(c);
	for (size_t i = 0, n = dims.size(); i < n; ++i)
	{
		// contract reverses left, right arguments
		dims[i] = {dims[i].second, dims[i].first};
	}
	if (is_2d(a.shape()) && is_2d(b.shape()) &&
		dims.size() == 1 && dims[0].first == 1 && dims[0].second == 0)
	{
		return make_eigenmatrix<T,Eigen::Product<MatMapT<T>,MatMapT<T>>,
			std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap(a.data(), a.shape()),
				make_matmap(b.data(), b.shape())},
			[](std::vector<MatMapT<T>>& args)
			{
				return args[0] * args[1];
			});
	}
	DimensionsT outdims = shape_convert(outshape);
	std::vector<TensMapT<T>> args = {
		make_tensmap(a.data(), a.shape()),
		make_tensmap(b.data(), b.shape())};
	_ARRAY_SWITCH(dims, _EIGEN_MATMUL_CASE);
}

#undef _EIGEN_MATMUL_CASE

/// Apply convolution of kernel across input
template <typename T>
EigenptrT<T> convolution (teq::Shape outshape, const iEigenEdge<T>& input,
	const iEigenEdge<T>& kernel, const marsh::Maps& attrs)
{
	std::vector<teq::RankT> order;
	auto c = get_coorder(attrs);
	if (c.empty())
	{
		logs::fatal("cannot convolve tensors without specified dimensions");
	}

	bool visited[teq::rank_cap];
	std::fill(visited, visited + teq::rank_cap, false);
	size_t n = std::min(c.size(), (size_t) teq::rank_cap);
	for (size_t i = 0; i < n; ++i)
	{
		teq::RankT d = c[i];
		if (visited[d])
		{
			logs::fatalf("convolution does not support repeated kernel dimensions: %s",
				fmts::to_string(c.begin(), c.end()).c_str());
		}
		visited[d] = true;
		order.push_back(d);
	}
	auto kshape = kernel.shape();
	for (size_t i = n; i < teq::rank_cap; ++i)
	{
		if (kshape.at(i) > 1)
		{
			logs::fatalf("given kernel shape %s, unspecified "
				"non-singular kernel dimension %d is undefined",
				kshape.to_string().c_str(), i);
		}
	}
	for (teq::RankT i = 0; i < teq::rank_cap; ++i)
	{
		if (visited[i] == false)
		{
			order.push_back(i);
		}
	}
	teq::ShapeT dims;
	std::copy(order.begin(), order.end(), dims.begin());
	return make_eigentensor<T,Eigen::TensorConvolutionOp<
		const teq::ShapeT,
		const TensMapT<T>,const TensMapT<T>>,
		std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap(input.data(), input.shape()),
			make_tensmap(kernel.data(), kernel.shape())},
		[&](std::vector<TensMapT<T>>& args)
		{
			return args[0].convolve(args[1], dims);
		});
}

#undef _ARRAY_SWITCH

}

#endif // EIGEN_OPERATOR_HPP
