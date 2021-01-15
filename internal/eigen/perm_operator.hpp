///
/// operator.hpp
/// eigen
///
/// Purpose:
/// Define functions manipulating tensor data values
/// No function in this file makes any attempt to check for nullptrs
///
// todo: make this generated

#ifdef PERM_OP
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
	case 0: global::fatal("missing dimensions");\
	case 1: CASE(ARR,1)\
	case 2: CASE(ARR,2)\
	case 3: CASE(ARR,3)\
	case 4: CASE(ARR,4)\
	case 5: CASE(ARR,5)\
	case 6: CASE(ARR,6)\
	case 7: CASE(ARR,7)\
	default: break;\
} CASE(ARR,8)

#define _EIGEN_RSUM_CASE(ARR, N)\
return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&in},\
[ARR,outdims](TensorT<T>& out, const std::vector<TensMapT<T>>& args){\
	out = args[0].sum(::eigen::internal::dim_copy<N>(ARR)).reshape(outdims);\
});

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

#define _EIGEN_RPROD_CASE(ARR, N)\
return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&in},\
[ARR,outdims](TensorT<T>& out, const std::vector<TensMapT<T>>& args){\
	out = args[0].prod(::eigen::internal::dim_copy<N>(ARR)).reshape(outdims);\
});

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

#define _EIGEN_RMIN_CASE(ARR, N)\
return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&in},\
[ARR,outdims](TensorT<T>& out, const std::vector<TensMapT<T>>& args){\
	out = args[0].minimum(::eigen::internal::dim_copy<N>(ARR)).reshape(outdims);\
});

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

#define _EIGEN_RMAX_CASE(ARR, N)\
return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&in},\
[ARR,outdims](TensorT<T>& out, const std::vector<TensMapT<T>>& args){\
	out = args[0].maximum(::eigen::internal::dim_copy<N>(ARR)).reshape(outdims);\
});

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

/// Return Eigen data object that argmax in tensor at return_dim
template <typename T>
EigenptrT argmax (teq::Shape outshape, const teq::iTensor& in, const marsh::iAttributed& attrib)
{
	teq::RankT return_dim;
	Packer<teq::RankT>().unpack(return_dim, attrib);

	DimensionsT outdims = shape_convert(outshape);
	if (return_dim >= teq::rank_cap)
	{
		return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&in},
		[outdims](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
		{
			out = args[0].argmax().template cast<T>().reshape(outdims);
		});
	}
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&in},
	[return_dim,outdims](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].argmax(return_dim).template cast<T>().reshape(outdims);
	});
}

/// Return Eigen data object representing data broadcast across dimensions
template <typename T>
EigenptrT extend (teq::Shape outshape, const teq::iTensor& in, const marsh::iAttributed& attrib)
{
	teq::Shape inshape = in.shape();
	teq::DimsT bcast = *unpack_extend(inshape, attrib);

	teq::ShapeT coord;
	std::fill(coord.begin(), coord.end(), 1);
	std::copy(bcast.begin(), bcast.begin() +
		std::min((size_t) teq::rank_cap, bcast.size()), coord.begin());
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&in},
	[coord](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].broadcast(coord);
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
		// use matrix when possible
		return std::make_shared<PermMatOp<T>>(outshape,teq::CTensT{&in},
		[](MatrixT<T>& out, const std::vector<MatMapT<T>>& args)
		{
			out = args[0].transpose();
		});
	}
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&in},
	[reorder](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].shuffle(reorder);
	});
}

/// Return Eigen data object representing data slicing of dimensions
template <typename T>
EigenptrT slice (teq::Shape outshape, const teq::TensptrT& in, const marsh::iAttributed& attrib)
{
	PairVecT<teq::DimT> encoding;
	Packer<PairVecT<teq::DimT>>().unpack(encoding, attrib);

	teq::Shape shape = in->shape();
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
		auto incr = index * batchsize;
		if (incr == 0)
		{
			return std::make_shared<TensRef>(*in);
		}
		return std::make_shared<UnsafeTensRef<T>>(*in, index * batchsize);
	}
	DimensionsT outdims = shape_convert(outshape);
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{in.get()},
	[offsets,extents,outdims](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].slice(offsets, extents).reshape(outdims);
	});
}

/// Return Eigen data object representing data zero padding
template <typename T>
EigenptrT pad (teq::Shape outshape, const teq::iTensor& in, const marsh::iAttributed& attrib)
{
	PairVecT<teq::DimT> encoding;
	Packer<PairVecT<teq::DimT>>().unpack(encoding, attrib);

	std::array<std::pair<teq::DimT,teq::DimT>,teq::rank_cap> paddings;
	std::fill(paddings.begin(), paddings.end(),
		std::pair<teq::DimT,teq::DimT>{0, 0});
	std::copy(encoding.begin(), encoding.begin() +
		std::min((size_t) teq::rank_cap, encoding.size()),
		paddings.begin());
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&in},
	[paddings](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].pad(paddings);
	});
}

/// Return Eigen data object representing strided view of in
template <typename T>
EigenptrT stride (teq::Shape outshape, const teq::iTensor& in, const marsh::iAttributed& attrib)
{
	teq::DimsT c;
	Packer<teq::DimsT>().unpack(c, attrib);

	Eigen::array<Eigen::DenseIndex,teq::rank_cap> incrs;
	std::fill(incrs.begin(), incrs.end(), 1);
	std::copy(c.begin(), c.begin() + std::min((size_t) teq::rank_cap, c.size()),
		incrs.begin());
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&in},
	[incrs](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].stride(incrs);
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
	return std::make_shared<PermTensOp<T>>(outshape, teq::CTensT{&in},
	[incrs](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out.setZero();
		out.stride(incrs) = args[0];
	});
}

template <typename T>
EigenptrT reverse (teq::Shape outshape, const teq::iTensor& in, const marsh::iAttributed& attrib)
{
	std::set<teq::RankT> dims;
	Packer<std::set<teq::RankT>>().unpack(dims, attrib);

	std::array<bool,teq::rank_cap> do_reverse;
	std::fill(do_reverse.begin(), do_reverse.end(), false);
	for (teq::RankT i : dims)
	{
		do_reverse[i] = true;
	}
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&in},
	[do_reverse](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].reverse(do_reverse);
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
		return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{group[0].get(),group[1].get()},
		[axis](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
		{
			out = args[0].concatenate(args[1],axis);
		});
	}
	teq::CTensT args;
	args.reserve(group.size());
	std::transform(group.begin(), group.end(), std::back_inserter(args),
	[](teq::TensptrT arg)
	{
		return arg.get();
	});
	std::array<Eigen::Index,teq::rank_cap-1> reshaped;
	auto it = outshape.begin();
	std::copy(it, it + axis, reshaped.begin());
	std::copy(it + axis + 1, outshape.end(), reshaped.begin() + axis);
	return std::make_shared<PermTensOp<T>>(outshape, args,
	[axis,reshaped](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		for (size_t i = 0, n = args.size(); i < n; ++i)
		{
			out.chip(i,axis) = args[i].reshape(reshaped);
		}
	});
}

/// Given reference to output array, and input vector ref,
/// make output elements referencing input tensor
EigenptrT ref (const teq::TensptrT& in);

/// Given reference to output array, and input vector ref,
/// make output elements take absolute value of inputs
template <typename T>
EigenptrT abs (teq::Shape outshape, const teq::iTensor& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return std::make_shared<PermMatOp<T>>(outshape,teq::CTensT{&in},
		[](MatrixT<T>& out, const std::vector<MatMapT<T>>& args)
		{
			out = args[0].cwiseAbs();
		});
	}
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&in},
	[](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].abs();
	});
}

/// Given reference to output array, and input vector ref,
/// make output elements take negatives of inputs
template <typename T>
EigenptrT neg (teq::Shape outshape, const teq::iTensor& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return std::make_shared<PermMatOp<T>>(outshape,teq::CTensT{&in},
		[](MatrixT<T>& out, const std::vector<MatMapT<T>>& args)
		{
			out = -args[0];
		});
	}
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&in},
	[](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = -args[0];
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
			// use matrix when possible
			return std::make_shared<PermMatOp<T>>(outshape,teq::CTensT{&in},
			[](MatrixT<T>& out, const std::vector<MatMapT<T>>& args)
			{
				out = args[0].array().sin();
			});
		}
	}
#endif
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&in},
	[](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].unaryExpr(std::function<T(const T&)>(
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
			// use matrix when possible
			return std::make_shared<PermMatOp<T>>(outshape,teq::CTensT{&in},
			[](MatrixT<T>& out, const std::vector<MatMapT<T>>& args)
			{
				out = args[0].array().cos();
			});
		}
	}
#endif
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&in},
	[](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].unaryExpr(std::function<T(const T&)>(
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
		// use matrix when possible
		return std::make_shared<PermMatOp<T>>(outshape,teq::CTensT{&in},
		[](MatrixT<T>& out, const std::vector<MatMapT<T>>& args)
		{
			out = args[0].array().tan();
		});
	}
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&in},
	[](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].unaryExpr(std::function<T(const T&)>(
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
		// use matrix when possible
		return std::make_shared<PermMatOp<T>>(outshape,teq::CTensT{&in},
		[](MatrixT<T>& out, const std::vector<MatMapT<T>>& args)
		{
			out = args[0].array().exp();
		});
	}
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&in},
	[](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].exp();
	});
}

/// Given reference to output array, and input vector ref,
/// make output elements take natural log of inputs
template <typename T>
EigenptrT log (teq::Shape outshape, const teq::iTensor& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return std::make_shared<PermMatOp<T>>(outshape,teq::CTensT{&in},
		[](MatrixT<T>& out, const std::vector<MatMapT<T>>& args)
		{
			out = args[0].array().log();
		});
	}
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&in},
	[](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].log();
	});
}

/// Given reference to output array, and input vector ref,
/// make output elements take square root of inputs
template <typename T>
EigenptrT sqrt (teq::Shape outshape, const teq::iTensor& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return std::make_shared<PermMatOp<T>>(outshape,teq::CTensT{&in},
		[](MatrixT<T>& out, const std::vector<MatMapT<T>>& args)
		{
			out = args[0].cwiseSqrt();
		});
	}
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&in},
	[](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].sqrt();
	});
}

/// Given reference to output array, and input vector ref,
/// make output elements take rounded values of inputs
template <typename T>
EigenptrT round (teq::Shape outshape, const teq::iTensor& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return std::make_shared<PermMatOp<T>>(outshape,teq::CTensT{&in},
		[](MatrixT<T>& out, const std::vector<MatMapT<T>>& args)
		{
			out = args[0].array().round();
		});
	}
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&in},
	[](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].round();
	});
}

template <typename T>
EigenptrT sigmoid (teq::Shape outshape, const teq::iTensor& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return std::make_shared<PermMatOp<T>>(outshape,teq::CTensT{&in},
		[](MatrixT<T>& out, const std::vector<MatMapT<T>>& args)
		{
			out = args[0].unaryExpr(Eigen::internal::scalar_sigmoid_op<T>());
		});
	}
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&in},
	[](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].sigmoid();
	});
}

template <typename T>
EigenptrT tanh (teq::Shape outshape, const teq::iTensor& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return std::make_shared<PermMatOp<T>>(outshape,teq::CTensT{&in},
		[](MatrixT<T>& out, const std::vector<MatMapT<T>>& args)
		{
			out = args[0].unaryExpr(Eigen::internal::scalar_tanh_op<T>());
		});
	}
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&in},
	[](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].tanh();
	});
}

template <typename T>
EigenptrT square (teq::Shape outshape, const teq::iTensor& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return std::make_shared<PermMatOp<T>>(outshape,teq::CTensT{&in},
		[](MatrixT<T>& out, const std::vector<MatMapT<T>>& args)
		{
			out = args[0].unaryExpr(Eigen::internal::scalar_square_op<T>());
		});
	}
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&in},
	[](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].square();
	});
}

template <typename T>
EigenptrT cube (teq::Shape outshape, const teq::iTensor& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return std::make_shared<PermMatOp<T>>(outshape,teq::CTensT{&in},
		[](MatrixT<T>& out, const std::vector<MatMapT<T>>& args)
		{
			out = args[0].unaryExpr(Eigen::internal::scalar_cube_op<T>());
		});
	}
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&in},
	[](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].cube();
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
		// use matrix when possible
		return std::make_shared<PermMatOp<T>>(outshape,teq::CTensT{&a,&b},
		[](MatrixT<T>& out, const std::vector<MatMapT<T>>& args)
		{
			out = args[0].binaryExpr(args[1],
			std::function<T(const T&,const T&)>(
			[](const T& a, const T& b) -> T
			{
				return std::pow(a, b);
			}));
		});
	}
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&a,&b},
	[](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].binaryExpr(args[1],
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
			// use matrix when possible
			return std::make_shared<PermMatOp<T>>(outshape,teq::CTensT{a.get(),b.get()},
			[](MatrixT<T>& out, const std::vector<MatMapT<T>>& args)
			{
				out = args[0] + args[1];
			});
		}
		return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{a.get(),b.get()},
		[](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
		{
			out = args[0] + args[1];
		});
	}
	teq::CTensT args;
	args.reserve(group.size());
	std::transform(group.begin(), group.end(), std::back_inserter(args),
	[](teq::TensptrT arg)
	{
		return arg.get();
	});
	return std::make_shared<PermTensOp<T>>(outshape,args,
	[](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0];
		for (size_t i = 1, n = args.size(); i < n; ++i)
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
		// use matrix when possible
		return std::make_shared<PermMatOp<T>>(outshape,teq::CTensT{&a,&b},
		[](MatrixT<T>& out, const std::vector<MatMapT<T>>& args)
		{
			out = args[0] - args[1];
		});
	}
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&a,&b},
	[](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0] - args[1];
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
			// use matrix when possible
			return std::make_shared<PermMatOp<T>>(outshape,teq::CTensT{a.get(),b.get()},
			[](MatrixT<T>& out, const std::vector<MatMapT<T>>& args)
			{
				out = args[0].cwiseProduct(args[1]);
			});
		}
		return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{a.get(),b.get()},
		[](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
		{
			out = args[0] * args[1];
		});
	}
	teq::CTensT args;
	args.reserve(group.size());
	std::transform(group.begin(), group.end(), std::back_inserter(args),
	[](teq::TensptrT arg)
	{
		return arg.get();
	});
	return std::make_shared<PermTensOp<T>>(outshape,args,
	[outshape](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0];
		for (size_t i = 1, n = args.size(); i < n; ++i)
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
		// use matrix when possible
		return std::make_shared<PermMatOp<T>>(outshape,teq::CTensT{&a,&b},
		[](MatrixT<T>& out, const std::vector<MatMapT<T>>& args)
		{
			out = args[0].cwiseQuotient(args[1]);
		});
	}
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&a,&b},
	[](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0] / args[1];
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
		// use matrix when possible
		return std::make_shared<PermMatOp<T>>(outshape,teq::CTensT{&a,&b},
		[](MatrixT<T>& out, const std::vector<MatMapT<T>>& args)
		{
			out = args[0].binaryExpr(args[1],
			std::function<T(const T&,const T&)>(
			[](const T& a, const T& b) -> T
			{
				return a == b;
			}));
		});
	}
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&a,&b},
	[](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].binaryExpr(args[1],
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
		// use matrix when possible
		return std::make_shared<PermMatOp<T>>(outshape,teq::CTensT{&a,&b},
		[](MatrixT<T>& out, const std::vector<MatMapT<T>>& args)
		{
			out = args[0].binaryExpr(args[1],
			std::function<T(const T&,const T&)>(
			[](const T& a, const T& b) -> T
			{
				return a != b;
			}));
		});
	}
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&a,&b},
	[](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].binaryExpr(args[1],
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
		// use matrix when possible
		return std::make_shared<PermMatOp<T>>(outshape,teq::CTensT{&a,&b},
		[](MatrixT<T>& out, const std::vector<MatMapT<T>>& args)
		{
			out = args[0].binaryExpr(args[1],
			std::function<T(const T&,const T&)>(
			[](const T& a, const T& b) -> T
			{
				return a < b;
			}));
		});
	}
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&a,&b},
	[](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].binaryExpr(args[1],
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
		// use matrix when possible
		return std::make_shared<PermMatOp<T>>(outshape,teq::CTensT{&a,&b},
		[](MatrixT<T>& out, const std::vector<MatMapT<T>>& args)
		{
			out = args[0].binaryExpr(args[1],
			std::function<T(const T&,const T&)>(
			[](const T& a, const T& b) -> T
			{
				return a > b;
			}));
		});
	}
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&a,&b},
	[](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].binaryExpr(args[1],
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
		// use matrix when possible
		return std::make_shared<PermMatOp<T>>(outshape,teq::CTensT{&a,&b},
		[](MatrixT<T>& out, const std::vector<MatMapT<T>>& args)
		{
			out = args[0].cwiseMin(args[1]);
		});
	}
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&a,&b},
	[](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].cwiseMin(args[1]);
	});
}

/// Given arguments, for every mapped index i in range [0:max_nelems],
/// take the maximum all elements for all arguments
template <typename T>
EigenptrT max (teq::Shape outshape, const teq::iTensor& a, const teq::iTensor& b)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return std::make_shared<PermMatOp<T>>(outshape,teq::CTensT{&a,&b},
		[](MatrixT<T>& out, const std::vector<MatMapT<T>>& args)
		{
			out = args[0].cwiseMax(args[1]);
		});
	}
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&a,&b},
	[](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].cwiseMax(args[1]);
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
		// use matrix when possible
		return std::make_shared<PermMatOp<T>>(outshape,teq::CTensT{&a,&b},
		[](MatrixT<T>& out, const std::vector<MatMapT<T>>& args)
		{
			auto generator = global::get_generator();
			out = args[0].binaryExpr(args[1],
			std::function<T(const T&,const T&)>(
			[&generator](const T& a, const T& b)
			{ return generator->unif_int(a, b); }));
		});
	}
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&a,&b},
	[](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		auto generator = global::get_generator();
		out = args[0].binaryExpr(args[1],
		std::function<T(const T&,const T&)>(
		[&generator](const T& a, const T& b)
		{ return generator->unif_int(a, b); }));
	});
}

template <typename T, typename std::enable_if<!std::is_integral<T>::value>::type* = nullptr>
EigenptrT rand_uniform (teq::Shape outshape, const teq::iTensor& a, const teq::iTensor& b)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return std::make_shared<PermMatOp<T>>(outshape,teq::CTensT{&a,&b},
		[](MatrixT<T>& out, const std::vector<MatMapT<T>>& args)
		{
			auto generator = global::get_generator();
			out = args[0].binaryExpr(args[1],
			std::function<T(const T&,const T&)>(
			[&generator](const T& a, const T& b)
			{ return generator->unif_dec(a, b); }));
		});
	}
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&a,&b},
	[](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		auto generator = global::get_generator();
		out = args[0].binaryExpr(args[1],
		std::function<T(const T&,const T&)>(
		[&generator](const T& a, const T& b)
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
		// use matrix when possible
		return std::make_shared<PermMatOp<T>>(outshape,teq::CTensT{&condition,&then,&otherwise},
		[](MatrixT<T>& out, const std::vector<MatMapT<T>>& args)
		{
			out = args[0].select(args[1], args[2]);
		});
	}
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&condition,&then,&otherwise},
	[](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].select(args[1],args[2]);
	});
}

#define _EIGEN_MATMUL_CASE(ARR, N)\
return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&a,&b},\
[ARR,outdims](TensorT<T>& out, const std::vector<TensMapT<T>>& args){\
	out = args[1].contract(args[0], internal::dim_copy<N>(ARR)).reshape(outdims);\
});

/// Only applies to 2-d tensors
/// Apply matrix multiplication of a and b
template <typename T>
EigenptrT contract (teq::Shape outshape, const teq::iTensor& a, const teq::iTensor& b, const marsh::iAttributed& attrib)
{
	PairVecT<teq::RankT> dims;
	Packer<PairVecT<teq::RankT>>().unpack(dims, attrib);

	for (auto& d : dims)
	{
		// contract reverses left, right arguments
		std::swap(d.first, d.second);
	}
	auto ashape = a.shape();
	auto bshape = b.shape();
	if (is_2d(ashape) && is_2d(bshape) &&
		dims.size() == 1 && dims[0].first == 1 && dims[0].second == 0)
	{
		return std::make_shared<PermMatOp<T>>(outshape,teq::CTensT{&a,&b},
		[](MatrixT<T>& out, const std::vector<MatMapT<T>>& args)
		{
			out = args[0] * args[1];
		});
	}
	DimensionsT outdims = shape_convert(outshape);
	_ARRAY_SWITCH(dims, _EIGEN_MATMUL_CASE);
}

#undef _EIGEN_MATMUL_CASE

#undef _ARRAY_SWITCH

template <typename T>
EigenptrT matmul (teq::Shape outshape, const teq::iTensor& a, const teq::iTensor& b)
{
	auto ashape = a.shape();
	auto bshape = b.shape();
	if (!is_2d(ashape) && !is_2d(bshape))
	{
		teq::Shape os({outshape.at(0), outshape.at(1)});
		teq::Shape as({ashape.at(0), ashape.at(1)});
		teq::Shape bs({bshape.at(0), bshape.at(1)});
		size_t nbatches = outshape.n_elems() / os.n_elems();
		return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&a,&b},
		[nbatches,os,as,bs](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
		{
			auto odata = out.data();
			auto adata = args[0].data();
			auto bdata = args[1].data();
			size_t osize = os.n_elems(),
				asize = as.n_elems(),
				bsize = bs.n_elems();
			for (size_t i = 0; i < nbatches; ++i)
			{
				make_matmap(odata + i * osize, os) =
					make_matmap(adata + i * asize, as) *
					make_matmap(bdata + i * bsize, bs);
			}
		});
	}
	marsh::Maps contract_attr;
	PairVecT<teq::RankT> dims = {std::pair<teq::RankT,teq::RankT>{1, 0}};
	Packer<PairVecT<teq::RankT>>().unpack(dims, contract_attr);
	return contract<T>(outshape, a, b, contract_attr);
}

/// Apply convolution of kernel across input
template <typename T>
EigenptrT convolution (teq::Shape outshape, const teq::iTensor& input,
	const teq::iTensor& kernel, const marsh::iAttributed& attrib)
{
	auto kshape = kernel.shape();
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
	return std::make_shared<PermTensOp<T>>(outshape,teq::CTensT{&input,&kernel},
	[dims](TensorT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].convolve(args[1],dims);
	});
}

template <typename T>
EigenptrT assign (teq::iTensor& target, const teq::iTensor& source)
{
	return std::make_shared<TensAssign<T>>(target, source,
	[](TensMapT<T>& target, const TensMapT<T>& source)
	{
		target = source;
	});
}

template <typename T>
EigenptrT assign_add (teq::iTensor& target, const teq::iTensor& source)
{
	return std::make_shared<TensAssign<T>>(target, source,
	[](TensMapT<T>& target, const TensMapT<T>& source)
	{
		target += source;
	});
}

template <typename T>
EigenptrT assign_sub (teq::iTensor& target, const teq::iTensor& source)
{
	return std::make_shared<TensAssign<T>>(target, source,
	[](TensMapT<T>& target, const TensMapT<T>& source)
	{
		target -= source;
	});
}

template <typename T>
EigenptrT assign_mul (teq::iTensor& target, const teq::iTensor& source)
{
	return std::make_shared<TensAssign<T>>(target, source,
	[](TensMapT<T>& target, const TensMapT<T>& source)
	{
		target *= source;
	});
}

template <typename T>
EigenptrT assign_div (teq::iTensor& target, const teq::iTensor& source)
{
	return std::make_shared<TensAssign<T>>(target, source,
	[](TensMapT<T>& target, const TensMapT<T>& source)
	{
		target /= source;
	});
}

#define _EIGEN_CAST_CASE(INTYPE)\
return std::make_shared<PermTensOp<T,INTYPE>>(input->shape(),teq::CTensT{input.get()},\
[](TensorT<T>& out, const std::vector<TensMapT<INTYPE>>& args){\
	out = args[0].template cast<T>();\
});

/// Convert tensor from one type to specified template type
template <typename T>
EigenptrT cast (const teq::TensptrT& input)
{
	auto intype = (egen::_GENERATED_DTYPE) input->get_meta().type_code();
	if (egen::get_type<T>() == intype)
	{
		global::warnf("pointless to convert the same type %s",
			egen::name_type(intype).c_str());
		return std::make_shared<TensRef>(*input);
	}

	EigenptrT out;
	TYPE_LOOKUP(_EIGEN_CAST_CASE, intype);
	return out;
}

#undef _EIGEN_CAST_CASE

}

#endif // EIGEN_OPERATOR_HPP
#endif // PERM_OP
