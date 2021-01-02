///
/// operator.hpp
/// eigen
///
/// Purpose:
/// Define functions manipulating tensor data values
/// No function in this file makes any attempt to check for nullptrs
///
// todo: make this generated

#ifndef EIGEN_OPERATOR_HPP
#define EIGEN_OPERATOR_HPP

#include "internal/eigen/device.hpp"
#include "internal/eigen/packattr.hpp"

namespace eigen
{

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
	case 0: global::fatal("missing dimensions"); [[fallthrough]];\
	case 1: CASE(ARR,1)\
	case 2: CASE(ARR,2)\
	case 3: CASE(ARR,3)\
	case 4: CASE(ARR,4)\
	case 5: CASE(ARR,5)\
	case 6: CASE(ARR,6)\
	case 7: CASE(ARR,7)\
	default: break;\
} CASE(ARR,8)

template <typename T, size_t N>
using RsumRetT = Eigen::TensorReshapingOp<const DimensionsT,const ReduceOutT<Eigen::internal::SumReducer<T>,N,T>>;

#define _EIGEN_RSUM_CASE(ARR, N)\
return make_eigentensor<T,RsumRetT<T,N>,TensMapT<T>>(\
outdims, make_tensmap((T*) in.device().data(), in.shape()),\
[&ARR, &outdims](TensMapT<T>& in) -> RsumRetT<T,N> {\
return in.sum(::eigen::internal::dim_copy<N>(ARR)).reshape(outdims); });

/// Return Eigen data object representing reduction where aggregation is sum
template <typename T>
EigenptrT reduce_sum (teq::Shape outshape, const teq::iTensor& in, const marsh::iAttributed& attrib)
{
	std::set<teq::RankT> ranks;
	Packer<std::set<teq::RankT>>().unpack(ranks, attrib);
	teq::RanksT vranks(ranks.begin(), ranks.end());

	DimensionsT outdims = shape_convert(outshape);
	_ARRAY_SWITCH(vranks, _EIGEN_RSUM_CASE)
}

#undef _EIGEN_RSUM_CASE

template <typename T, size_t N>
using RprodRetT = Eigen::TensorReshapingOp<const DimensionsT,const ReduceOutT<Eigen::internal::ProdReducer<T>,N,T>>;

#define _EIGEN_RPROD_CASE(ARR, N)\
return make_eigentensor<T,RprodRetT<T,N>,TensMapT<T>>(\
outdims, make_tensmap((T*) in.device().data(), in.shape()),\
[&ARR, &outdims](TensMapT<T>& in) -> RprodRetT<T,N> {\
return in.prod(::eigen::internal::dim_copy<N>(ARR)).reshape(outdims); });

/// Return Eigen data object representing reduction where aggregation is prod
template <typename T>
EigenptrT reduce_prod (teq::Shape outshape, const teq::iTensor& in, const marsh::iAttributed& attrib)
{
	std::set<teq::RankT> ranks;
	Packer<std::set<teq::RankT>>().unpack(ranks, attrib);
	teq::RanksT vranks(ranks.begin(), ranks.end());

	DimensionsT outdims = shape_convert(outshape);
	_ARRAY_SWITCH(vranks, _EIGEN_RPROD_CASE)
}

#undef _EIGEN_RPROD_CASE

template <typename T, size_t N>
using RminRetT = Eigen::TensorReshapingOp<const DimensionsT,const ReduceOutT<Eigen::internal::MinReducer<T>,N,T>>;

#define _EIGEN_RMIN_CASE(ARR, N)\
return make_eigentensor<T,RminRetT<T,N>,TensMapT<T>>(\
outdims, make_tensmap((T*) in.device().data(), in.shape()),\
[&ARR, &outdims](TensMapT<T>& in) -> RminRetT<T,N> {\
return in.minimum(::eigen::internal::dim_copy<N>(ARR)).reshape(outdims); });

/// Return Eigen data object representing reduction where aggregation is min
template <typename T>
EigenptrT reduce_min (teq::Shape outshape, const teq::iTensor& in, const marsh::iAttributed& attrib)
{
	std::set<teq::RankT> ranks;
	Packer<std::set<teq::RankT>>().unpack(ranks, attrib);
	teq::RanksT vranks(ranks.begin(), ranks.end());

	DimensionsT outdims = shape_convert(outshape);
	_ARRAY_SWITCH(vranks, _EIGEN_RMIN_CASE)
}

#undef _EIGEN_RMIN_CASE

template <typename T, size_t N>
using RmaxRetT = Eigen::TensorReshapingOp<const DimensionsT,const ReduceOutT<Eigen::internal::MaxReducer<T>,N,T>>;

#define _EIGEN_RMAX_CASE(ARR, N)\
return make_eigentensor<T,RmaxRetT<T,N>,TensMapT<T>>(\
outdims, make_tensmap((T*) in.device().data(), in.shape()),\
[&ARR, &outdims](TensMapT<T>& in) -> RmaxRetT<T,N> {\
return in.maximum(::eigen::internal::dim_copy<N>(ARR)).reshape(outdims); });

/// Return Eigen data object representing reduction where aggregation is max
template <typename T>
EigenptrT reduce_max (teq::Shape outshape, const teq::iTensor& in, const marsh::iAttributed& attrib)
{
	std::set<teq::RankT> ranks;
	Packer<std::set<teq::RankT>>().unpack(ranks, attrib);
	teq::RanksT vranks(ranks.begin(), ranks.end());

	DimensionsT outdims = shape_convert(outshape);
	_ARRAY_SWITCH(vranks, _EIGEN_RMAX_CASE)
}

#undef _EIGEN_RMAX_CASE

template <typename T, size_t N>
using ArgmaxRetT = Eigen::TensorReshapingOp<const DimensionsT,const Eigen::TensorConversionOp<T,
	const Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<
	Eigen::Tuple<Eigen::Index,T>>,const Eigen::array<Eigen::Index,N>,const TensMapT<T>>>>;

/// Return Eigen data object that rgmax in tensor at return_dim
template <typename T>
EigenptrT argmax (teq::Shape outshape, const teq::iTensor& in, const marsh::iAttributed& attrib)
{
	teq::RankT return_dim;
	Packer<teq::RankT>().unpack(return_dim, attrib);

	DimensionsT outdims = shape_convert(outshape);
	if (return_dim >= teq::rank_cap)
	{
		return make_eigentensor<T,ArgmaxRetT<T,teq::rank_cap>,TensMapT<T>>(
			outdims, make_tensmap((T*) in.device().data(), in.shape()),
			[&outdims](TensMapT<T>& in) -> ArgmaxRetT<T,teq::rank_cap>
			{
				ArgmaxRetT<T,teq::rank_cap> out = in.argmax().template cast<T>().reshape(outdims);
				return out;
			});
	}
	return make_eigentensor<T,ArgmaxRetT<T,1>,TensMapT<T>>(
		outdims, make_tensmap((T*) in.device().data(), in.shape()),
		[return_dim, &outdims](TensMapT<T>& in) -> ArgmaxRetT<T,1>
		{
			return in.argmax(return_dim).template cast<T>().reshape(outdims);
		});
}

/// Return Eigen data object representing data broadcast across dimensions
template <typename T>
EigenptrT extend (teq::Shape outshape, const teq::iTensor& in, const marsh::iAttributed& attrib)
{
	using ExtendRetT = Eigen::TensorBroadcastingOp<const teq::ShapeT,const TensMapT<T>>;

	teq::Shape inshape = in.shape();
	teq::DimsT bcast = *unpack_extend(inshape, attrib);

	teq::ShapeT coord;
	std::fill(coord.begin(), coord.end(), 1);
	std::copy(bcast.begin(), bcast.begin() +
		std::min((size_t) teq::rank_cap, bcast.size()), coord.begin());
	return make_eigentensor<T,ExtendRetT,TensMapT<T>>(
		shape_convert(outshape), make_tensmap((T*) in.device().data(), inshape),
		[coord](TensMapT<T>& in) -> ExtendRetT
		{
			return in.broadcast(coord);
		});
}

/// Return Eigen data object representing transpose and permutation
template <typename T>
EigenptrT permute (teq::Shape outshape, const teq::iTensor& in, const marsh::iAttributed& attrib)
{
	teq::RanksT order;
	Packer<teq::RanksT>().unpack(order, attrib);

	bool visited[teq::rank_cap];
	std::fill(visited, visited + teq::rank_cap, false);
	for (teq::RankT i = 0, n = order.size(); i < n; ++i)
	{
		visited[order[i]] = true;
	}
	for (teq::RankT i = 0; i < teq::rank_cap; ++i)
	{
		if (false == visited[i])
		{
			order.push_back(i);
		}
	}
	auto reorder = internal::dim_copy<teq::rank_cap,teq::RankT>(order);
	if (is_2d(outshape) && reorder[0] == 1 && reorder[1] == 0)
	{
		using TransposeRetT = Eigen::Transpose<MatMapT<T>>;

		// use matrix when possible
		return make_eigenmatrix<T,TransposeRetT,MatMapT<T>>(
			shape_convert(outshape), make_matmap((T*) in.device().data(), in.shape()),
			[](MatMapT<T>& in) -> TransposeRetT
			{
				return in.transpose();
			});
	}
	using PermuteRetT = Eigen::TensorShufflingOp<const std::array<teq::RankT,teq::rank_cap>,TensMapT<T>>;	

	return make_eigentensor<T,PermuteRetT,TensMapT<T>>(
		shape_convert(outshape), make_tensmap((T*) in.device().data(), in.shape()),
		[reorder](TensMapT<T>& in) -> PermuteRetT
		{
			return in.shuffle(reorder);
		});
}

/// Return Eigen data object that reshapes
template <typename T>
EigenptrT reshape (teq::Shape outshape, const teq::iTensor& in)
{
	return make_eigentensor<T,TensMapT<T>,TensMapT<T>>(
		shape_convert(outshape), make_tensmap((T*) in.device().data(), in.shape()),
		[](TensMapT<T>& in) -> TensMapT<T> { return in; });
}

/// Return Eigen data object representing data slicing of dimensions
template <typename T>
EigenptrT slice (teq::Shape outshape, const teq::iTensor& in, const marsh::iAttributed& attrib)
{
	using SliceRetT = Eigen::TensorReshapingOp<const DimensionsT,
		Eigen::TensorSlicingOp<const teq::ShapeT, const teq::ShapeT,TensMapT<T>>>;

	PairVecT<teq::DimT> encoding;
	Packer<PairVecT<teq::DimT>>().unpack(encoding, attrib);

	teq::Shape shape = in.shape();
	teq::ShapeT offsets;
	teq::ShapeT extents;
	std::fill(offsets.begin(), offsets.end(), 0);
	std::copy(shape.begin(), shape.end(), extents.begin());
	size_t n = std::min(encoding.size(), (size_t) teq::rank_cap);
	for (size_t i = 0; i < n; ++i)
	{
		teq::DimT offset = std::min(encoding[i].first, (teq::DimT) (shape.at(i) - 1));
		offsets[i] = offset;
		extents[i] = std::min(encoding[i].second, (teq::DimT) (shape.at(i) - offset));
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
		return std::make_shared<PtrRef<T>>(
			(T*) in.device().data() + index * batchsize);
	}
	DimensionsT outdims = shape_convert(outshape);
	return make_eigentensor<T,SliceRetT,TensMapT<T>>(
		outdims, make_tensmap((T*) in.device().data(), shape),
		[&offsets, &extents, &outdims](TensMapT<T>& in) -> SliceRetT
		{
			return in.slice(offsets, extents).reshape(outdims);
		});
}

/// Return Eigen data object representing data zero padding
template <typename T>
EigenptrT pad (teq::Shape outshape, const teq::iTensor& in, const marsh::iAttributed& attrib)
{
	using PadRetT = Eigen::TensorPaddingOp<const std::array<
		std::pair<teq::DimT,teq::DimT>,teq::rank_cap>,const TensMapT<T>>;

	PairVecT<teq::DimT> encoding;
	Packer<PairVecT<teq::DimT>>().unpack(encoding, attrib);

	std::array<std::pair<teq::DimT,teq::DimT>,teq::rank_cap> paddings;
	std::fill(paddings.begin(), paddings.end(),
		std::pair<teq::DimT,teq::DimT>{0, 0});
	std::copy(encoding.begin(), encoding.begin() +
		std::min((size_t) teq::rank_cap, encoding.size()),
		paddings.begin());
	return make_eigentensor<T,PadRetT,TensMapT<T>>(
		shape_convert(outshape), make_tensmap((T*) in.device().data(), in.shape()),
		[&paddings](TensMapT<T>& in) -> PadRetT
		{
			return in.pad(paddings);
		});
}

/// Return Eigen data object representing strided view of in
template <typename T>
EigenptrT stride (teq::Shape outshape, const teq::iTensor& in, const marsh::iAttributed& attrib)
{
	using StrideRetT = Eigen::TensorStridingOp<const Eigen::array<
		Eigen::DenseIndex,teq::rank_cap>,TensMapT<T>>;

	teq::DimsT c;
	Packer<teq::DimsT>().unpack(c, attrib);

	Eigen::array<Eigen::DenseIndex,teq::rank_cap> incrs;
	std::fill(incrs.begin(), incrs.end(), 1);
	std::copy(c.begin(), c.begin() + std::min((size_t) teq::rank_cap, c.size()),
		incrs.begin());
	return make_eigentensor<T,StrideRetT,TensMapT<T>>(
		shape_convert(outshape), make_tensmap((T*) in.device().data(), in.shape()),
		[&incrs](TensMapT<T>& in) -> StrideRetT
		{
			return in.stride(incrs);
		});
}

/// Return Eigen data object that scatters data in
/// specific increments across dimensions
/// This function is the reverse of stride
template <typename T>
EigenptrT scatter (teq::Shape outshape, const teq::iTensor& in, const marsh::iAttributed& attrib)
{
	teq::DimsT dims;
	Packer<teq::DimsT>().unpack(dims, attrib);

	Eigen::array<Eigen::DenseIndex,teq::rank_cap> incrs;
	std::fill(incrs.begin(), incrs.end(), 1);
	std::copy(dims.begin(), dims.begin() +
		std::min((size_t) teq::rank_cap, dims.size()), incrs.begin());
	return std::make_shared<TensAccum<T,TensMapT<T>>>(
		0, shape_convert(outshape), make_tensmap((T*) in.device().data(), in.shape()),
		[incrs](TensorT<T>& out, const TensMapT<T>& in)
		{
			out.stride(incrs) = in;
		});
}

template <typename T>
EigenptrT reverse (teq::Shape outshape, const teq::iTensor& in, const marsh::iAttributed& attrib)
{
	using ReverseRetT = Eigen::TensorReverseOp<const std::array<
		bool,teq::rank_cap>,TensMapT<T>>;

	std::set<teq::RankT> dims;
	Packer<std::set<teq::RankT>>().unpack(dims, attrib);

	std::array<bool,teq::rank_cap> do_reverse;
	std::fill(do_reverse.begin(), do_reverse.end(), false);
	for (teq::RankT i : dims)
	{
		do_reverse[i] = true;
	}
	return make_eigentensor<T,ReverseRetT,TensMapT<T>>(
		shape_convert(outshape), make_tensmap((T*) in.device().data(), in.shape()),
		[&do_reverse](TensMapT<T>& in) -> ReverseRetT
		{
			return in.reverse(do_reverse);
		});
}

template <typename T>
EigenptrT concat (teq::Shape outshape, const teq::TensptrsT& group, const marsh::iAttributed& attrib)
{
	assert(group.size() > 1);
	teq::RankT axis;
	Packer<teq::RankT>().unpack(axis, attrib);
	if (group.size() == 2)
	{
		using ConcatRetT = Eigen::TensorConcatenationOp<
			const teq::RankT,TensMapT<T>,TensMapT<T>>;

		const teq::TensptrT& left = group[0];
		const teq::TensptrT& right = group[1];
		return make_eigentensor<T,ConcatRetT,std::vector<TensMapT<T>>>(
			shape_convert(outshape), {
				make_tensmap((T*) left->device().data(), left->shape()),
				make_tensmap((T*) right->device().data(), right->shape())},
			[axis](std::vector<TensMapT<T>>& args) -> ConcatRetT
			{
				return args[0].concatenate(args[1], axis);
			});
	}
	std::vector<TensMapT<T>> args;
	args.reserve(group.size());
	std::transform(group.begin(), group.end(), std::back_inserter(args),
		[](teq::TensptrT arg)
		{
			return make_tensmap((T*) arg->device().data(), arg->shape());
		});
	std::array<Eigen::Index,teq::rank_cap-1> reshaped;
	auto it = outshape.begin();
	std::copy(it, it + axis, reshaped.begin());
	std::copy(it + axis + 1, outshape.end(), reshaped.begin() + axis);
	return std::make_shared<TensAccum<T,std::vector<TensMapT<T>>>>(
		0, shape_convert(outshape), args,
		[axis,reshaped](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
		{
			for (size_t i = 0, n = args.size(); i < n; ++i)
			{
				out.chip(i, axis) = args[i].reshape(reshaped);
			}
		});
}

/// Given reference to output array, and input vector ref,
/// make output elements referencing input tensor
template <typename T>
EigenptrT ref (teq::Shape, const teq::iTensor& in)
{
	return std::make_shared<PtrRef<T>>((T*) in.device().data());
}

/// Given reference to output array, and input vector ref,
/// make output elements take absolute value of inputs
template <typename T>
EigenptrT abs (teq::Shape outshape, const teq::iTensor& in)
{
	if (is_2d(outshape))
	{
		using AbsMatRetT = Eigen::CwiseUnaryOp<
			Eigen::internal::scalar_abs_op<T>,const MatMapT<T>>;

		// use matrix when possible
		return make_eigenmatrix<T,AbsMatRetT,MatMapT<T>>(shape_convert(outshape),
			make_matmap((T*) in.device().data(), in.shape()),
			[](MatMapT<T>& in) -> AbsMatRetT
			{
				return in.cwiseAbs();
			});
	}
	using AbsTensRetT = Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_abs_op<T>,const TensMapT<T>>;

	return make_eigentensor<T,AbsTensRetT,TensMapT<T>>(shape_convert(outshape),
		make_tensmap((T*) in.device().data(), in.shape()),
		[](TensMapT<T>& in) -> AbsTensRetT
		{
			return in.abs();
		});
}

/// Given reference to output array, and input vector ref,
/// make output elements take negatives of inputs
template <typename T>
EigenptrT neg (teq::Shape outshape, const teq::iTensor& in)
{
	if (is_2d(outshape))
	{
		using NegMatRetT = Eigen::CwiseUnaryOp<
			Eigen::internal::scalar_opposite_op<T>,const MatMapT<T>>;

		// use matrix when possible
		return make_eigenmatrix<T,NegMatRetT,MatMapT<T>>(shape_convert(outshape),
			make_matmap((T*) in.device().data(), in.shape()),
			[](MatMapT<T>& in) -> NegMatRetT
			{
				return -in;
			});
	}
	using NegTensRetT = Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_opposite_op<T>,const TensMapT<T>>;

	return make_eigentensor<T,NegTensRetT,TensMapT<T>>(shape_convert(outshape),
		make_tensmap((T*) in.device().data(), in.shape()),
		[](TensMapT<T>& in) -> NegTensRetT
		{
			return -in;
		});
}

/// Given reference to output array, and input vector ref,
/// make output elements take sine of inputs
template <typename T>
EigenptrT sin (teq::Shape outshape, const teq::iTensor& in)
{
#ifdef __cpp_if_constexpr
	if constexpr(!std::is_integral<T>::value)
	{
		if (is_2d(outshape))
		{
			using SinMatRetT = typename Eigen::ArrayWrapper<MatMapT<T>>::SinReturnType;

			// use matrix when possible
			return make_eigenmatrix<T,SinMatRetT,MatMapT<T>>(shape_convert(outshape),
				make_matmap((T*) in.device().data(), in.shape()),
				[](MatMapT<T>& in) -> SinMatRetT
				{
					return in.array().sin();
				});
		}
	}
#endif
	using SinTensRetT = Eigen::TensorCwiseUnaryOp<
		std::function<T(const T&)>,const TensMapT<T>>;

	return make_eigentensor<T,SinTensRetT,TensMapT<T>>(shape_convert(outshape),
		make_tensmap((T*) in.device().data(), in.shape()),
		[](TensMapT<T>& in) -> SinTensRetT
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
EigenptrT cos (teq::Shape outshape, const teq::iTensor& in)
{
#ifdef __cpp_if_constexpr
	if constexpr(!std::is_integral<T>::value)
	{
		if (is_2d(outshape))
		{
			using CosMatRetT = typename Eigen::ArrayWrapper<MatMapT<T>>::CosReturnType;

			// use matrix when possible
			return make_eigenmatrix<T,CosMatRetT,MatMapT<T>>(shape_convert(outshape),
				make_matmap((T*) in.device().data(), in.shape()),
				[](MatMapT<T>& in) -> CosMatRetT
				{
					return in.array().cos();
				});
		}
	}
#endif
	using CosTensRetT = Eigen::TensorCwiseUnaryOp<
		std::function<T(const T&)>,const TensMapT<T>>;

	return make_eigentensor<T,CosTensRetT,TensMapT<T>>(shape_convert(outshape),
		make_tensmap((T*) in.device().data(), in.shape()),
		[](TensMapT<T>& in) -> CosTensRetT
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
EigenptrT tan (teq::Shape outshape, const teq::iTensor& in)
{
	if (is_2d(outshape))
	{
		using TanMatRetT = typename Eigen::ArrayWrapper<MatMapT<T>>::TanReturnType;

		// use matrix when possible
		return make_eigenmatrix<T,TanMatRetT,MatMapT<T>>(shape_convert(outshape),
			make_matmap((T*) in.device().data(), in.shape()),
			[](MatMapT<T>& in) -> TanMatRetT
			{
				return in.array().tan();
			});
	}
	using TanTensRetT = Eigen::TensorCwiseUnaryOp<
		std::function<T(const T&)>,const TensMapT<T>>;

	return make_eigentensor<T,TanTensRetT,TensMapT<T>>(shape_convert(outshape),
		make_tensmap((T*) in.device().data(), in.shape()),
		[](TensMapT<T>& in) -> TanTensRetT
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
EigenptrT exp (teq::Shape outshape, const teq::iTensor& in)
{
	if (is_2d(outshape))
	{
		using ExpMatRetT = typename Eigen::ArrayWrapper<MatMapT<T>>::ExpReturnType;

		// use matrix when possible
		return make_eigenmatrix<T,ExpMatRetT,MatMapT<T>>(shape_convert(outshape),
			make_matmap((T*) in.device().data(), in.shape()),
			[](MatMapT<T>& in) -> ExpMatRetT
			{
				return in.array().exp();
			});
	}
	using ExpTensRetT = Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_exp_op<T>,const TensMapT<T>>;

	return make_eigentensor<T,ExpTensRetT,TensMapT<T>>(shape_convert(outshape),
		make_tensmap((T*) in.device().data(), in.shape()),
		[](TensMapT<T>& in) -> ExpTensRetT
		{
			return in.exp();
		});
}

/// Given reference to output array, and input vector ref,
/// make output elements take natural log of inputs
template <typename T>
EigenptrT log (teq::Shape outshape, const teq::iTensor& in)
{
	if (is_2d(outshape))
	{
		using LogMatRetT = typename Eigen::ArrayWrapper<MatMapT<T>>::LogReturnType;

		// use matrix when possible
		return make_eigenmatrix<T,LogMatRetT,MatMapT<T>>(shape_convert(outshape),
			make_matmap((T*) in.device().data(), in.shape()),
			[](MatMapT<T>& in) -> LogMatRetT
			{
				return in.array().log();
			});
	}
	using LogTensRetT = Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_log_op<T>,const TensMapT<T>>;

	return make_eigentensor<T,LogTensRetT,TensMapT<T>>(shape_convert(outshape),
		make_tensmap((T*) in.device().data(), in.shape()),
		[](TensMapT<T>& in) -> LogTensRetT
		{
			return in.log();
		});
}

/// Given reference to output array, and input vector ref,
/// make output elements take square root of inputs
template <typename T>
EigenptrT sqrt (teq::Shape outshape, const teq::iTensor& in)
{
	if (is_2d(outshape))
	{
		using SqrtMatRetT = Eigen::CwiseUnaryOp<
			Eigen::internal::scalar_sqrt_op<T>,const MatMapT<T>>;

		// use matrix when possible
		return make_eigenmatrix<T,SqrtMatRetT,MatMapT<T>>(shape_convert(outshape),
			make_matmap((T*) in.device().data(), in.shape()),
			[](MatMapT<T>& in) -> SqrtMatRetT
			{
				return in.cwiseSqrt();
			});
	}
	using SqrtTensRetT = Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_sqrt_op<T>,const TensMapT<T>>;

	return make_eigentensor<T,SqrtTensRetT,TensMapT<T>>(shape_convert(outshape),
		make_tensmap((T*) in.device().data(), in.shape()),
		[](TensMapT<T>& in) -> SqrtTensRetT
		{
			return in.sqrt();
		});
}

/// Given reference to output array, and input vector ref,
/// make output elements take rounded values of inputs
template <typename T>
EigenptrT round (teq::Shape outshape, const teq::iTensor& in)
{
	if (is_2d(outshape))
	{
		using RoundMatRetT = typename Eigen::ArrayWrapper<MatMapT<T>>::RoundReturnType;

		// use matrix when possible
		return make_eigenmatrix<T,RoundMatRetT,MatMapT<T>>(shape_convert(outshape),
			make_matmap((T*) in.device().data(), in.shape()),
			[](MatMapT<T>& in) -> RoundMatRetT
			{
				return in.array().round();
			});
	}
	using RoundTensRetT = Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_round_op<T>,const TensMapT<T>>;

	return make_eigentensor<T,RoundTensRetT,TensMapT<T>>(shape_convert(outshape),
		make_tensmap((T*) in.device().data(), in.shape()),
		[](TensMapT<T>& in) -> RoundTensRetT
		{
			return in.round();
		});
}

template <typename T>
EigenptrT sigmoid (teq::Shape outshape, const teq::iTensor& in)
{
	if (is_2d(outshape))
	{
		using SigmoidMatRetT = Eigen::CwiseUnaryOp<
			Eigen::internal::scalar_sigmoid_op<T>,const MatMapT<T>>;

		// use matrix when possible
		return make_eigenmatrix<T,SigmoidMatRetT,MatMapT<T>>(shape_convert(outshape),
			make_matmap((T*) in.device().data(), in.shape()),
			[](MatMapT<T>& in) -> SigmoidMatRetT
			{
				return in.unaryExpr(Eigen::internal::scalar_sigmoid_op<T>());
			});
	}
	using SigmoidTensRetT = Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_sigmoid_op<T>,const TensMapT<T>>;

	return make_eigentensor<T,SigmoidTensRetT,TensMapT<T>>(shape_convert(outshape),
		make_tensmap((T*) in.device().data(), in.shape()),
		[](TensMapT<T>& in) -> SigmoidTensRetT
		{
			return in.sigmoid();
		});
}

template <typename T>
EigenptrT tanh (teq::Shape outshape, const teq::iTensor& in)
{
	if (is_2d(outshape))
	{
		using TanhMatRetT = Eigen::CwiseUnaryOp<
			Eigen::internal::scalar_tanh_op<T>,const MatMapT<T>>;

		// use matrix when possible
		return make_eigenmatrix<T,TanhMatRetT,MatMapT<T>>(shape_convert(outshape),
			make_matmap((T*) in.device().data(), in.shape()),
			[](MatMapT<T>& in) -> TanhMatRetT
			{
				return in.unaryExpr(Eigen::internal::scalar_tanh_op<T>());
			});
	}
	using TanhTensRetT = Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_tanh_op<T>,const TensMapT<T>>;

	return make_eigentensor<T,TanhTensRetT,TensMapT<T>>(shape_convert(outshape),
		make_tensmap((T*) in.device().data(), in.shape()),
		[](TensMapT<T>& in) -> TanhTensRetT
		{
			return in.tanh();
		});
}

template <typename T>
EigenptrT square (teq::Shape outshape, const teq::iTensor& in)
{
	if (is_2d(outshape))
	{
		using SquareMatRetT = Eigen::CwiseUnaryOp<
			Eigen::internal::scalar_square_op<T>,const MatMapT<T>>;

		// use matrix when possible
		return make_eigenmatrix<T,SquareMatRetT,MatMapT<T>>(shape_convert(outshape),
			make_matmap((T*) in.device().data(), in.shape()),
			[](MatMapT<T>& in) -> SquareMatRetT
			{
				return in.unaryExpr(Eigen::internal::scalar_square_op<T>());
			});
	}
	using SquareTensRetT = Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_square_op<T>,const TensMapT<T>>;

	return make_eigentensor<T,SquareTensRetT,TensMapT<T>>(shape_convert(outshape),
		make_tensmap((T*) in.device().data(), in.shape()),
		[](TensMapT<T>& in) -> SquareTensRetT
		{
			return in.square();
		});
}

template <typename T>
EigenptrT cube (teq::Shape outshape, const teq::iTensor& in)
{
	if (is_2d(outshape))
	{
		using CubeMatRetT = Eigen::CwiseUnaryOp<
			Eigen::internal::scalar_cube_op<T>,const MatMapT<T>>;

		// use matrix when possible
		return make_eigenmatrix<T,CubeMatRetT,MatMapT<T>>(shape_convert(outshape),
			make_matmap((T*) in.device().data(), in.shape()),
			[](MatMapT<T>& in) -> CubeMatRetT
			{
				return in.unaryExpr(Eigen::internal::scalar_cube_op<T>());
			});
	}
	using CubeTensRetT = Eigen::TensorCwiseUnaryOp<
		Eigen::internal::scalar_cube_op<T>,const TensMapT<T>>;

	return make_eigentensor<T,CubeTensRetT,TensMapT<T>>(shape_convert(outshape),
		make_tensmap((T*) in.device().data(), in.shape()),
		[](TensMapT<T>& in) -> CubeTensRetT
		{
			return in.cube();
		});
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply std::pow operator
/// Only accept 2 arguments
template <typename T>
EigenptrT pow (teq::Shape outshape, const teq::iTensor& a, const teq::iTensor& b)
{
	if (is_2d(outshape))
	{
		using PowMatRetT = Eigen::CwiseBinaryOp<std::function<
			T(const T&,const T&)>,const MatMapT<T>,const MatMapT<T>>;

		// use matrix when possible
		return make_eigenmatrix<T,PowMatRetT,std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap((T*) a.device().data(), a.shape()),
				make_matmap((T*) b.device().data(), b.shape())},
			[](std::vector<MatMapT<T>>& args) -> PowMatRetT
			{
				return args[0].binaryExpr(args[1],
					std::function<T(const T&,const T&)>(
					[](const T& a, const T& b) -> T
					{
						return std::pow(a, b);
					}));
			});
	}
	using PowTensRetT = Eigen::TensorCwiseBinaryOp<std::function<
		T(const T&,const T&)>,const TensMapT<T>,const TensMapT<T>>;

	return make_eigentensor<T,PowTensRetT,std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap((T*) a.device().data(), a.shape()),
			make_tensmap((T*) b.device().data(), b.shape())},
		[](std::vector<TensMapT<T>>& args) -> PowTensRetT
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
EigenptrT add (teq::Shape outshape, const teq::TensptrsT& group)
{
	assert(group.size() > 1);
	if (group.size() == 2)
	{
		const teq::TensptrT& a = group[0];
		const teq::TensptrT& b = group[1];
		if (is_2d(outshape))
		{
			using AddMatRetT = Eigen::CwiseBinaryOp<
				Eigen::internal::scalar_sum_op<T>,const MatMapT<T>,const MatMapT<T>>;

			// use matrix when possible
			return make_eigenmatrix<T,AddMatRetT,std::vector<MatMapT<T>>>(shape_convert(outshape), {
					make_matmap((T*) a->device().data(), a->shape()),
					make_matmap((T*) b->device().data(), b->shape())},
				[](std::vector<MatMapT<T>>& args) -> AddMatRetT
				{
					return args[0] + args[1];
				});
		}
		using AddTensRetT = Eigen::TensorCwiseBinaryOp<
			Eigen::internal::scalar_sum_op<T>,const TensMapT<T>,const TensMapT<T>>;

		return make_eigentensor<T,AddTensRetT,std::vector<TensMapT<T>>>(shape_convert(outshape), {
				make_tensmap((T*) a->device().data(), a->shape()),
				make_tensmap((T*) b->device().data(), b->shape())},
			[](std::vector<TensMapT<T>>& args) -> AddTensRetT
			{
				return args[0] + args[1];
			});
	}
	std::vector<TensMapT<T>> args;
	args.reserve(group.size());
	std::transform(group.begin(), group.end(), std::back_inserter(args),
		[](teq::TensptrT arg)
		{
			return make_tensmap((T*) arg->device().data(), arg->shape());
		});
	return std::make_shared<TensAccum<T,std::vector<TensMapT<T>>>>(
		0, shape_convert(outshape), args,
		[](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
		{
			for (size_t i = 0, n = args.size(); i < n; ++i)
			{
				out += args[i];
			}
		});
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index subtract
/// Only accept 2 arguments
template <typename T>
EigenptrT sub (teq::Shape outshape, const teq::iTensor& a, const teq::iTensor& b)
{
	if (is_2d(outshape))
	{
		using SubMatRetT = Eigen::CwiseBinaryOp<
			Eigen::internal::scalar_difference_op<T>,const MatMapT<T>,const MatMapT<T>>;

		// use matrix when possible
		return make_eigenmatrix<T,SubMatRetT,std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap((T*) a.device().data(), a.shape()),
				make_matmap((T*) b.device().data(), b.shape())},
			[](std::vector<MatMapT<T>>& args) -> SubMatRetT
			{
				return args[0] - args[1];
			});
	}
	using SubTensRetT = Eigen::TensorCwiseBinaryOp<
		Eigen::internal::scalar_difference_op<T>,const TensMapT<T>,const TensMapT<T>>;

	return make_eigentensor<T,SubTensRetT,std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap((T*) a.device().data(), a.shape()),
			make_tensmap((T*) b.device().data(), b.shape())},
		[](std::vector<TensMapT<T>>& args) -> SubTensRetT
		{
			return args[0] - args[1];
		});
}

template <typename T>
EigenptrT mul (teq::Shape outshape, const teq::TensptrsT& group)
{
	assert(group.size() > 1);
	if (group.size() == 2)
	{
		const teq::TensptrT& a = group[0];
		const teq::TensptrT& b = group[1];
		if (is_2d(outshape))
		{
			using MulMatRetT = Eigen::CwiseBinaryOp<
				Eigen::internal::scalar_product_op<T>,const MatMapT<T>,const MatMapT<T>>;

			// use matrix when possible
			return make_eigenmatrix<T,MulMatRetT,std::vector<MatMapT<T>>>(shape_convert(outshape), {
					make_matmap((T*) a->device().data(), a->shape()),
					make_matmap((T*) b->device().data(), b->shape())},
				[](std::vector<MatMapT<T>>& args) -> MulMatRetT
				{
					return args[0].cwiseProduct(args[1]);
				});
		}
		using MulTensRetT = Eigen::TensorCwiseBinaryOp<
			Eigen::internal::scalar_product_op<T>,const TensMapT<T>,const TensMapT<T>>;

		return make_eigentensor<T,MulTensRetT,std::vector<TensMapT<T>>>(shape_convert(outshape), {
				make_tensmap((T*) a->device().data(), a->shape()),
				make_tensmap((T*) b->device().data(), b->shape())},
			[](std::vector<TensMapT<T>>& args) -> MulTensRetT
			{
				return args[0] * args[1];
			});
	}
	std::vector<TensMapT<T>> args;
	args.reserve(group.size());
	std::transform(group.begin(), group.end(), std::back_inserter(args),
		[](teq::TensptrT arg)
		{
			return make_tensmap((T*) arg->device().data(), arg->shape());
		});
	return std::make_shared<TensAccum<T,std::vector<TensMapT<T>>>>(
		1, shape_convert(outshape), args,
		[](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
		{
			for (size_t i = 0, n = args.size(); i < n; ++i)
			{
				out *= args[i];
			}
		});
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index divide
/// Only accept 2 arguments
template <typename T>
EigenptrT div (teq::Shape outshape, const teq::iTensor& a, const teq::iTensor& b)
{
	if (is_2d(outshape))
	{
		using DivMatRetT = Eigen::CwiseBinaryOp<
			Eigen::internal::scalar_quotient_op<T>,const MatMapT<T>,const MatMapT<T>>;

		// use matrix when possible
		return make_eigenmatrix<T,DivMatRetT,std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap((T*) a.device().data(), a.shape()),
				make_matmap((T*) b.device().data(), b.shape())},
			[](std::vector<MatMapT<T>>& args) -> DivMatRetT
			{
				return args[0].cwiseQuotient(args[1]);
			});
	}
	using DivTensRetT = Eigen::TensorCwiseBinaryOp<
		Eigen::internal::scalar_quotient_op<T>,const TensMapT<T>,const TensMapT<T>>;

	return make_eigentensor<T,DivTensRetT,std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap((T*) a.device().data(), a.shape()),
			make_tensmap((T*) b.device().data(), b.shape())},
		[](std::vector<TensMapT<T>>& args) -> DivTensRetT
		{
			return args[0] / args[1];
		});
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply == operator
/// Only accept 2 arguments
template <typename T>
EigenptrT eq (teq::Shape outshape, const teq::iTensor& a, const teq::iTensor& b)
{
	if (is_2d(outshape))
	{
		using EqMatRetT = Eigen::CwiseBinaryOp<std::function<
			T(const T&,const T&)>,const MatMapT<T>,const MatMapT<T>>;

		// use matrix when possible
		return make_eigenmatrix<T,EqMatRetT,std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap((T*) a.device().data(), a.shape()),
				make_matmap((T*) b.device().data(), b.shape())},
			[](std::vector<MatMapT<T>>& args) -> EqMatRetT
			{
				return args[0].binaryExpr(args[1],
					std::function<T(const T&,const T&)>(
					[](const T& a, const T& b) -> T
					{
						return a == b;
					}));
			});
	}
	using EqTensRetT = Eigen::TensorCwiseBinaryOp<std::function<
		T(const T&,const T&)>,const TensMapT<T>,const TensMapT<T>>;

	return make_eigentensor<T,EqTensRetT,std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap((T*) a.device().data(), a.shape()),
			make_tensmap((T*) b.device().data(), b.shape())},
		[](std::vector<TensMapT<T>>& args) -> EqTensRetT
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
EigenptrT neq (teq::Shape outshape, const teq::iTensor& a, const teq::iTensor& b)
{
	if (is_2d(outshape))
	{
		using NeqMatRetT = Eigen::CwiseBinaryOp<std::function<
			T(const T&,const T&)>,const MatMapT<T>,const MatMapT<T>>;

		// use matrix when possible
		return make_eigenmatrix<T,NeqMatRetT,std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap((T*) a.device().data(), a.shape()),
				make_matmap((T*) b.device().data(), b.shape())},
			[](std::vector<MatMapT<T>>& args) -> NeqMatRetT
			{
				return args[0].binaryExpr(args[1],
					std::function<T(const T&,const T&)>(
					[](const T& a, const T& b) -> T
					{
						return a != b;
					}));
			});
	}
	using NeqTensRetT = Eigen::TensorCwiseBinaryOp<std::function<
		T(const T&,const T&)>,const TensMapT<T>,const TensMapT<T>>;

	return make_eigentensor<T,NeqTensRetT,std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap((T*) a.device().data(), a.shape()),
			make_tensmap((T*) b.device().data(), b.shape())},
		[](std::vector<TensMapT<T>>& args) -> NeqTensRetT
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
EigenptrT lt (teq::Shape outshape, const teq::iTensor& a, const teq::iTensor& b)
{
	if (is_2d(outshape))
	{
		using LtMatRetT = Eigen::CwiseBinaryOp<std::function<
			T(const T&,const T&)>,const MatMapT<T>,const MatMapT<T>>;

		// use matrix when possible
		return make_eigenmatrix<T,LtMatRetT,std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap((T*) a.device().data(), a.shape()),
				make_matmap((T*) b.device().data(), b.shape())},
			[](std::vector<MatMapT<T>>& args) -> LtMatRetT
			{
				return args[0].binaryExpr(args[1],
					std::function<T(const T&,const T&)>(
					[](const T& a, const T& b) -> T
					{
						return a < b;
					}));
			});
	}
	using LtTensRetT = Eigen::TensorCwiseBinaryOp<std::function<
		T(const T&,const T&)>,const TensMapT<T>,const TensMapT<T>>;

	return make_eigentensor<T,LtTensRetT,std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap((T*) a.device().data(), a.shape()),
			make_tensmap((T*) b.device().data(), b.shape())},
		[](std::vector<TensMapT<T>>& args) -> LtTensRetT
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
EigenptrT gt (teq::Shape outshape, const teq::iTensor& a, const teq::iTensor& b)
{
	if (is_2d(outshape))
	{
		using GtMatRetT = Eigen::CwiseBinaryOp<std::function<
			T(const T&,const T&)>,const MatMapT<T>,const MatMapT<T>>;

		// use matrix when possible
		return make_eigenmatrix<T,GtMatRetT,std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap((T*) a.device().data(), a.shape()),
				make_matmap((T*) b.device().data(), b.shape())},
			[](std::vector<MatMapT<T>>& args) -> GtMatRetT
			{
				return args[0].binaryExpr(args[1],
					std::function<T(const T&,const T&)>(
					[](const T& a, const T& b) -> T
					{
						return a > b;
					}));
			});
	}
	using GtTensRetT = Eigen::TensorCwiseBinaryOp<std::function<
		T(const T&,const T&)>,const TensMapT<T>,const TensMapT<T>>;

	return make_eigentensor<T,GtTensRetT,std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap((T*) a.device().data(), a.shape()),
			make_tensmap((T*) b.device().data(), b.shape())},
		[](std::vector<TensMapT<T>>& args) -> GtTensRetT
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
EigenptrT min (teq::Shape outshape, const teq::iTensor& a, const teq::iTensor& b)
{
	if (is_2d(outshape))
	{
		using MinMatRetT = Eigen::CwiseBinaryOp<
			Eigen::internal::scalar_min_op<T,T>,const MatMapT<T>,const MatMapT<T>>;

		// use matrix when possible
		return make_eigenmatrix<T,MinMatRetT,std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap((T*) a.device().data(), a.shape()),
				make_matmap((T*) b.device().data(), b.shape())},
			[](std::vector<MatMapT<T>>& args) -> MinMatRetT
			{
				return args[0].cwiseMin(args[1]);
			});
	}
	using MinTensRetT = Eigen::TensorCwiseBinaryOp<
		Eigen::internal::scalar_min_op<T>,const TensMapT<T>,const TensMapT<T>>;

	return make_eigentensor<T,MinTensRetT,std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap((T*) a.device().data(), a.shape()),
			make_tensmap((T*) b.device().data(), b.shape())},
		[](std::vector<TensMapT<T>>& args) -> MinTensRetT
		{
			return args[0].cwiseMin(args[1]);
		});
}

/// Given arguments, for every mapped index i in range [0:max_nelems],
/// take the maximum all elements for all arguments
template <typename T>
EigenptrT max (teq::Shape outshape, const teq::iTensor& a, const teq::iTensor& b)
{
	if (is_2d(outshape))
	{
		using MaxMatRetT = Eigen::CwiseBinaryOp<
			Eigen::internal::scalar_max_op<T,T>,const MatMapT<T>,const MatMapT<T>>;

		// use matrix when possible
		return make_eigenmatrix<T,MaxMatRetT,std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap((T*) a.device().data(), a.shape()),
				make_matmap((T*) b.device().data(), b.shape())},
			[](std::vector<MatMapT<T>>& args) -> MaxMatRetT
			{
				return args[0].cwiseMax(args[1]);
			});
	}
	using MaxTensRetT = Eigen::TensorCwiseBinaryOp<
		Eigen::internal::scalar_max_op<T>,const TensMapT<T>,const TensMapT<T>>;

	return make_eigentensor<T,MaxTensRetT,std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap((T*) a.device().data(), a.shape()),
			make_tensmap((T*) b.device().data(), b.shape())},
		[](std::vector<TensMapT<T>>& args) -> MaxTensRetT
		{
			return args[0].cwiseMax(args[1]);
		});
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply std::uniform_distributon function
/// Only accept 2 arguments
template <typename T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
EigenptrT rand_uniform (teq::Shape outshape, const teq::iTensor& a, const teq::iTensor& b)
{
	if (is_2d(outshape))
	{
		using RUnifIntMatRetT = Eigen::CwiseBinaryOp<std::function<
			T(const T&,const T&)>,const MatMapT<T>,const MatMapT<T>>;

		// use matrix when possible
		return make_eigenmatrix<T,RUnifIntMatRetT,std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap((T*) a.device().data(), a.shape()),
				make_matmap((T*) b.device().data(), b.shape())},
			[](std::vector<MatMapT<T>>& args) -> RUnifIntMatRetT
			{
				auto generator = global::get_generator();
				return args[0].binaryExpr(args[1],
					std::function<T(const T&,const T&)>(
						[generator](const T& a, const T& b)
						{ return generator->unif_int(a, b); }));
			});
	}
	using RUnifIntTensRetT = Eigen::TensorCwiseBinaryOp<std::function<
		T(const T&,const T&)>,const TensMapT<T>,const TensMapT<T>>;

	return make_eigentensor<T,RUnifIntTensRetT,std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap((T*) a.device().data(), a.shape()),
			make_tensmap((T*) b.device().data(), b.shape())},
		[](std::vector<TensMapT<T>>& args) -> RUnifIntTensRetT
		{
			auto generator = global::get_generator();
			return args[0].binaryExpr(args[1],
				std::function<T(const T&,const T&)>(
					[generator](const T& a, const T& b)
					{ return generator->unif_int(a, b); }));
		});
}

template <typename T, typename std::enable_if<!std::is_integral<T>::value>::type* = nullptr>
EigenptrT rand_uniform (teq::Shape outshape, const teq::iTensor& a, const teq::iTensor& b)
{
	if (is_2d(outshape))
	{
		using RUnifDecMatRetT = Eigen::CwiseBinaryOp<
			std::function<T(const T&,const T&)>,
			const MatMapT<T>,const MatMapT<T>>;

		// use matrix when possible
		return make_eigenmatrix<T,RUnifDecMatRetT,std::vector<MatMapT<T>>>(shape_convert(outshape), {
				make_matmap((T*) a.device().data(), a.shape()),
				make_matmap((T*) b.device().data(), b.shape())},
			[](std::vector<MatMapT<T>>& args) -> RUnifDecMatRetT
			{
				auto generator = global::get_generator();
				return args[0].binaryExpr(args[1],
					std::function<T(const T&,const T&)>(
						[generator](const T& a, const T& b)
						{ return generator->unif_dec(a, b); }));
			});
	}
	using RUnifDecTensRetT = Eigen::TensorCwiseBinaryOp<
		std::function<T(const T&,const T&)>,const TensMapT<T>,const TensMapT<T>>;

	return make_eigentensor<T,RUnifDecTensRetT,std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap((T*) a.device().data(), a.shape()),
			make_tensmap((T*) b.device().data(), b.shape())},
		[](std::vector<TensMapT<T>>& args) -> RUnifDecTensRetT
		{
			auto generator = global::get_generator();
			return args[0].binaryExpr(args[1],
				std::function<T(const T&,const T&)>(
					[generator](const T& a, const T& b)
					{ return generator->unif_dec(a, b); }));
		});
}

/// Given a condition, then values and otherwise
/// apply corresponding then value if condition is non-zero
/// otherwise apply otherwise value
template <typename T>
EigenptrT select (teq::Shape outshape, const teq::iTensor& condition,
	const teq::iTensor& then, const teq::iTensor& otherwise)
{
	if (is_2d(outshape))
	{
		using SelectMatRetT = Eigen::Select<MatMapT<T>,MatMapT<T>,MatMapT<T>>;

		// use matrix when possible
		return make_eigenmatrix<T,SelectMatRetT,std::vector<MatMapT<T>>>(
			shape_convert(outshape), {
			make_matmap((T*) condition.device().data(), condition.shape()),
			make_matmap((T*) then.device().data(), then.shape()),
			make_matmap((T*) otherwise.device().data(), otherwise.shape())},
			[](std::vector<MatMapT<T>>& args) -> SelectMatRetT
			{
				return args[0].select(args[1], args[2]);
			});
	}
	using SelectTensRetT = Eigen::TensorSelectOp<const TensMapT<T>,
		const TensMapT<T>,const TensMapT<T>>;

	return make_eigentensor<T,SelectTensRetT,std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap((T*) condition.device().data(), condition.shape()),
			make_tensmap((T*) then.device().data(), then.shape()),
			make_tensmap((T*) otherwise.device().data(), otherwise.shape())},
		[](std::vector<TensMapT<T>>& args) -> SelectTensRetT
		{
			return args[0].select(args[1], args[2]);
		});
}

template <typename T, size_t N>
using ContractionRetT = Eigen::TensorReshapingOp<const DimensionsT,
	const Eigen::TensorContractionOp<const std::array<
	std::pair<teq::RankT,teq::RankT>,N>,const TensMapT<T>,const TensMapT<T>>>;

#define _EIGEN_MATMUL_CASE(ARR, N)\
return make_eigentensor<T,ContractionRetT<T,N>,std::vector<TensMapT<T>>>(outdims, args,\
[&](std::vector<TensMapT<T>>& args) -> ContractionRetT<T,N> {\
return args[1].contract(args[0], internal::dim_copy<N>(ARR)).reshape(outdims); });

/// Only applies to 2-d tensors
/// Apply matrix multiplication of a and b
template <typename T>
EigenptrT matmul (teq::Shape outshape, const teq::iTensor& a, const teq::iTensor& b, const marsh::iAttributed& attrib)
{
	PairVecT<teq::RankT> dims;
	Packer<PairVecT<teq::RankT>>().unpack(dims, attrib);

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
				make_matmap((T*) a.device().data(), a.shape()),
				make_matmap((T*) b.device().data(), b.shape())},
			[](std::vector<MatMapT<T>>& args) -> Eigen::Product<MatMapT<T>,MatMapT<T>>
			{
				return args[0] * args[1];
			});
	}
	DimensionsT outdims = shape_convert(outshape);
	std::vector<TensMapT<T>> args = {
		make_tensmap((T*) a.device().data(), a.shape()),
		make_tensmap((T*) b.device().data(), b.shape())};
	_ARRAY_SWITCH(dims, _EIGEN_MATMUL_CASE);
}

#undef _EIGEN_MATMUL_CASE

#undef _ARRAY_SWITCH

/// Apply convolution of kernel across input
template <typename T>
EigenptrT convolution (teq::Shape outshape, const teq::iTensor& input,
	const teq::iTensor& kernel, const marsh::iAttributed& attrib)
{
	using ConvRetT = Eigen::TensorConvolutionOp<const teq::ShapeT,
		const TensMapT<T>,const TensMapT<T>>;

	teq::RanksT order;
	Packer<teq::RanksT>().unpack(order, attrib);

	bool visited[teq::rank_cap];
	std::fill(visited, visited + teq::rank_cap, false);
	size_t n = std::min(order.size(), (size_t) teq::rank_cap);
	for (size_t i = 0; i < n; ++i)
	{
		teq::RankT d = order[i];
		if (visited[d])
		{
			global::fatalf("convolution does not support repeated kernel "
				"dimensions: %s", fmts::to_string(
					order.begin(), order.end()).c_str());
		}
		visited[d] = true;
	}
	auto kshape = kernel.shape();
	for (size_t i = n; i < teq::rank_cap; ++i)
	{
		if (kshape.at(i) > 1)
		{
			global::fatalf("given kernel shape %s, unspecified "
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
	return make_eigentensor<T,ConvRetT,std::vector<TensMapT<T>>>(shape_convert(outshape), {
			make_tensmap((T*) input.device().data(), input.shape()),
			make_tensmap((T*) kernel.device().data(), kernel.shape())},
		[&](std::vector<TensMapT<T>>& args)
		{
			return args[0].convolve(args[1], dims);
		});
}

template <typename T>
EigenptrT assign (teq::iTensor& target, const teq::iTensor& source)
{
	return std::make_shared<TensAssign<T>>(target, source,
		[&target](TensorT<T>& target_tens, const teq::iTensor& source)
		{
			static_cast<iMutableLeaf&>(target).upversion(
				source.get_meta().state_version() + 1);
			auto data = (T*) source.device().data();
			target_tens = make_tensmap(data, source.shape());
		});
}

template <typename T>
EigenptrT assign_add (teq::iTensor& target, const teq::iTensor& source)
{
	return std::make_shared<TensAssign<T>>(target, source,
		[&target](TensorT<T>& target_tens, const teq::iTensor& source)
		{
			static_cast<iMutableLeaf&>(target).upversion(
				source.get_meta().state_version() + 1);
			auto data = (T*) source.device().data();
			target_tens += make_tensmap(data, source.shape());
		});
}

template <typename T>
EigenptrT assign_sub (teq::iTensor& target, const teq::iTensor& source)
{
	return std::make_shared<TensAssign<T>>(target, source,
		[&target](TensorT<T>& target_tens, const teq::iTensor& source)
		{
			static_cast<iMutableLeaf&>(target).upversion(
				source.get_meta().state_version() + 1);
			auto data = (T*) source.device().data();
			target_tens -= make_tensmap(data, source.shape());
		});
}

template <typename T>
EigenptrT assign_mul (teq::iTensor& target, const teq::iTensor& source)
{
	return std::make_shared<TensAssign<T>>(target, source,
		[&target](TensorT<T>& target_tens, const teq::iTensor& source)
		{
			static_cast<iMutableLeaf&>(target).upversion(
				source.get_meta().state_version() + 1);
			auto data = (T*) source.device().data();
			target_tens *= make_tensmap(data, source.shape());
		});
}

template <typename T>
EigenptrT assign_div (teq::iTensor& target, const teq::iTensor& source)
{
	return std::make_shared<TensAssign<T>>(target, source,
		[&target](TensorT<T>& target_tens, const teq::iTensor& source)
		{
			static_cast<iMutableLeaf&>(target).upversion(
				source.get_meta().state_version() + 1);
			auto data = (T*) source.device().data();
			target_tens /= make_tensmap(data, source.shape());
		});
}

#define _EIGEN_CAST_CASE(INTYPE)\
out = make_eigentensor<T,Eigen::TensorConversionOp<T,const TensMapT<INTYPE>>,TensMapT<INTYPE>>(\
shape_convert(input.shape()),make_tensmap(\
(INTYPE*) input.device().data(), input.shape()),\
[&](TensMapT<INTYPE>& arg) -> Eigen::TensorConversionOp<T,const TensMapT<INTYPE>> {\
	return arg.template cast<T>();\
});

/// Convert tensor from one type to specified template type
template <typename T>
EigenptrT cast (const teq::iTensor& input)
{
	auto intype = (egen::_GENERATED_DTYPE) input.get_meta().type_code();
	if (egen::get_type<T>() == intype)
	{
		global::warnf("pointless to convert the same type %s",
			egen::name_type(intype).c_str());
		return std::make_shared<PtrRef<T>>((T*) input.device().data());
	}

	EigenptrT out;
	TYPE_LOOKUP(_EIGEN_CAST_CASE, intype);
	return out;
}

#undef _EIGEN_CAST_CASE

}

#endif // EIGEN_OPERATOR_HPP
