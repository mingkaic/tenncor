///
/// operator.hpp
/// eigen
///
/// Purpose:
/// Define functions manipulating tensor data values
/// No function in this file makes any attempt to check for nullptrs
///
// todo: make this generated

#ifndef PERM_OP
#ifndef EIGEN_OPERATOR_HPP
#define EIGEN_OPERATOR_HPP

#include "internal/eigen/device.hpp"
#include "internal/eigen/packattr.hpp"

namespace eigen
{

template <typename T>
using BinaryF = std::function<T(const T&,const T&)>;

static bool is_2d (teq::Shape shape)
{
	return std::all_of(shape.begin() + 2, shape.end(),
	[](teq::DimT dim) { return 1 == dim; });
}

namespace internal
{

/// Return array of input vector
template <size_t N, typename T=teq::RankT>
static std::array<T,N> dim_copy (const std::vector<T>& d)
{
	std::array<T,N> out;
	auto it = d.begin();
	std::copy(it, it + N, out.begin());
	return out;
}

template <typename T>
using UnaryMatOpF = std::function<void(MatMapT<T>&,const MatMapT<T>&)>;

template <typename T>
using UnarySMatOpF = std::function<SMatrixT<T>(const SMatMapT<T>&)>;

template <typename T>
static EigenptrT unary_smatop (
	const teq::Shape& outshape, const teq::iTensor& arg,
	UnarySMatOpF<T> sparse_op, UnaryMatOpF<T> norm_op)
{
	if (is_sparse(arg))
	{
		return std::make_shared<SparseMatOp<T>>(teq::CTensT{&arg},
		[sparse_op](const teq::CTensT& args)
		{
			auto smat = make_smatmap<T>(*args.front());
			return sparse_op(smat.get());
		});
	}
	return std::make_shared<MatOp<T>>(outshape, teq::CTensT{&arg},
	[norm_op](MatMapT<T>& out, const std::vector<MatMapT<T>>& args)
	{
		norm_op(out, args.front());
	});
}

template <typename T>
static EigenptrT unary_matop (
	const teq::Shape& outshape, const teq::iTensor& arg,
	typename GenericMatOp<T>::OpF sparse_op, UnaryMatOpF<T> norm_op)
{
	if (is_sparse(arg))
	{
		return std::make_shared<GenericMatOp<T>>(outshape, teq::CTensT{&arg}, sparse_op);
	}
	return std::make_shared<MatOp<T>>(outshape, teq::CTensT{&arg},
	[norm_op](MatMapT<T>& out, const std::vector<MatMapT<T>>& args)
	{
		norm_op(out, args.front());
	});
}

// does the same as unary_matop, except convert sparse mat to mat base
template <typename T>
static EigenptrT unary_matop_smat2mat (const teq::Shape& outshape,
	const teq::iTensor& arg, UnaryMatOpF<T> op)
{
	return unary_matop<T>(outshape, arg,
	[op](MatMapT<T>& out, const teq::CTensT& args)
	{
		MatrixT<T> mat = make_smatmap<T>(*args.front()).get();
		op(out, mat_to_matmap<T>(mat));
	}, op);
}

template <typename T>
static EigenptrT nnary_matop (const teq::Shape& outshape, const teq::CTensT& args,
	std::function<size_t(MatMapT<T>&,const teq::iTensor&)> init,
	std::function<size_t(MatMapT<T>&,const teq::iTensor&,size_t)> sparse_accum,
	std::function<size_t(MatMapT<T>&,const teq::iTensor&,size_t)> norm_accum)
{
	return std::make_shared<GenericMatOp<T>>(outshape, args,
	[init, sparse_accum, norm_accum](MatMapT<T>& out, const teq::CTensT& args)
	{
		auto& arg = args.front();
		size_t iter = init(out, *arg);
		for (size_t i = 1, n = args.size(); i < n; ++i)
		{
			if (is_sparse(*args[i]))
			{
				iter = sparse_accum(out, *args[i], iter);
			}
			else
			{
				iter = norm_accum(out, *args[i], iter);
			}
		}
	});
}

template <typename T>
static EigenptrT binaryexpr_matop (const teq::Shape& outshape,
	const teq::iTensor& lhs, const teq::iTensor& rhs,
	BinaryF<T> func)
{
	auto lsparse = is_sparse(lhs);
	auto rsparse = is_sparse(rhs);
	if (lsparse && rsparse)
	{
		return std::make_shared<GenericMatOp<T>>(outshape, teq::CTensT{&lhs, &rhs},
		[func](MatMapT<T>& out, const teq::CTensT& args)
		{
			out = make_smatmap<T>(*args.front())->binaryExpr(
				make_smatmap<T>(*args.back()).get(), func);
		});
	}
	else if (lsparse && !rsparse)
	{
		return std::make_shared<GenericMatOp<T>>(outshape, teq::CTensT{&lhs, &rhs},
		[func](MatMapT<T>& out, const teq::CTensT& args)
		{
			MatrixT<T> am = make_smatmap<T>(*args.front()).get();
			out = am.binaryExpr(make_matmap<T>(*args.back()).get(), func);
		});
	}
	else if (!lsparse && rsparse)
	{
		return std::make_shared<GenericMatOp<T>>(outshape, teq::CTensT{&lhs, &rhs},
		[func](MatMapT<T>& out, const teq::CTensT& args)
		{
			MatrixT<T> bm = make_smatmap<T>(*args.back()).get();
			out = make_matmap<T>(*args.front())->binaryExpr(bm, func);
		});
	}
	else
	{
		return std::make_shared<GenericMatOp<T>>(outshape, teq::CTensT{&lhs, &rhs},
		[func](MatMapT<T>& out, const teq::CTensT& args)
		{
			out = make_matmap<T>(*args.front())->binaryExpr(
				make_matmap<T>(*args.back()).get(), func);
		});
	}
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

#define _EIGEN_RSUM_CASE(ARR, N)\
return std::make_shared<TensOp<T>>(outshape,teq::CTensT{in.get()},\
[outdims,ARR](TensMapT<T>& out, const std::vector<TensMapT<T>>& args){\
	out = args[0].sum(::eigen::internal::dim_copy<N>(ARR)).reshape(outdims);\
});

/// Return Eigen data object representing reduction where aggregation is sum
template <typename T>
EigenptrT reduce_sum (teq::Shape outshape, const teq::TensptrT& in, const marsh::iAttributed& attrib)
{
	DimensionsT outdims = shape_convert(outshape);
	std::set<teq::RankT> ranks;
	Packer<std::set<teq::RankT>>().unpack(ranks, attrib);

	auto inshape = in->shape();
	if (is_2d(inshape))
	{
		auto rowwise = estd::has(ranks, 0);
		auto colwise = estd::has(ranks, 1);
		auto ncols = inshape.at(0);
		auto nrows = inshape.at(1);
		if (colwise && rowwise)
		{
			return internal::unary_matop<T>(outshape, *in,
			[](MatMapT<T>& out, const teq::CTensT& args)
			{
				*(out.data()) = make_smatmap<T>(*args.front())->sum();
			},
			[](MatMapT<T>& out, const MatMapT<T>& arg)
			{
				*(out.data()) = arg.sum();
			});
		}
		else if (colwise)
		{
			return internal::unary_matop<T>(outshape, *in,
			[nrows](MatMapT<T>& out, const teq::CTensT& args)
			{
				out = MatrixT<T>::Ones(1, nrows) * make_smatmap<T>(*args.front()).get();
			},
			[](MatMapT<T>& out, const MatMapT<T>& arg)
			{
				out = arg.colwise().sum();
			});
		}
		else if (rowwise)
		{
			return internal::unary_matop<T>(outshape, *in,
			[ncols](MatMapT<T>& out, const teq::CTensT& args)
			{
				out = make_smatmap<T>(*args.front()).get() * MatrixT<T>::Ones(ncols, 1);
			},
			[](MatMapT<T>& out, const MatMapT<T>& arg)
			{
				out = arg.rowwise().sum();
			});
		}
		else
		{
			return std::make_shared<TensRef>(*in);
		}
	}

	teq::RanksT vranks(ranks.begin(), ranks.end());
	_ARRAY_SWITCH(vranks, _EIGEN_RSUM_CASE)
}

#undef _EIGEN_RSUM_CASE

#define _MATRIX_REDUCE(REDUCE_OP)\
if (is_2d(in->shape())){\
	auto rowwise = estd::has(ranks, 0);\
	auto colwise = estd::has(ranks, 1);\
	if (colwise && rowwise){\
		return internal::unary_matop_smat2mat<T>(outshape,*in,\
		[](MatMapT<T>& out, const MatMapT<T>& arg){\
			*(out.data()) = arg.REDUCE_OP();\
		});\
	}else if (colwise){\
		return internal::unary_matop_smat2mat<T>(outshape,*in,\
		[](MatMapT<T>& out, const MatMapT<T>& arg){\
			out = arg.colwise().REDUCE_OP();\
		});\
	}else if (rowwise){\
		return internal::unary_matop_smat2mat<T>(outshape,*in,\
		[](MatMapT<T>& out, const MatMapT<T>& arg){\
			out = arg.rowwise().REDUCE_OP();\
		});\
	}else{\
		return std::make_shared<TensRef>(*in);\
	}\
}

#define _EIGEN_RPROD_CASE(ARR, N)\
return std::make_shared<TensOp<T>>(outshape,teq::CTensT{in.get()},\
[outdims,ARR](TensMapT<T>& out, const std::vector<TensMapT<T>>& args){\
	out = args[0].prod(::eigen::internal::dim_copy<N>(ARR)).reshape(outdims);\
});

/// Return Eigen data object representing reduction where aggregation is prod
template <typename T>
EigenptrT reduce_prod (teq::Shape outshape, const teq::TensptrT& in, const marsh::iAttributed& attrib)
{
	std::set<teq::RankT> ranks;
	Packer<std::set<teq::RankT>>().unpack(ranks, attrib);

	_MATRIX_REDUCE(prod);

	teq::RanksT vranks(ranks.begin(), ranks.end());

	DimensionsT outdims = shape_convert(outshape);
	_ARRAY_SWITCH(vranks, _EIGEN_RPROD_CASE)
}

#undef _EIGEN_RPROD_CASE

#define _EIGEN_RMIN_CASE(ARR, N)\
return std::make_shared<TensOp<T>>(outshape,teq::CTensT{in.get()},\
[outdims,ARR](TensMapT<T>& out, const std::vector<TensMapT<T>>& args){\
	out = args[0].minimum(::eigen::internal::dim_copy<N>(ARR)).reshape(outdims);\
});

/// Return Eigen data object representing reduction where aggregation is min
template <typename T>
EigenptrT reduce_min (teq::Shape outshape, const teq::TensptrT& in, const marsh::iAttributed& attrib)
{
	std::set<teq::RankT> ranks;
	Packer<std::set<teq::RankT>>().unpack(ranks, attrib);

	_MATRIX_REDUCE(minCoeff);

	teq::RanksT vranks(ranks.begin(), ranks.end());

	DimensionsT outdims = shape_convert(outshape);
	_ARRAY_SWITCH(vranks, _EIGEN_RMIN_CASE)
}

#undef _EIGEN_RMIN_CASE

#define _EIGEN_RMAX_CASE(ARR, N)\
return std::make_shared<TensOp<T>>(outshape,teq::CTensT{in.get()},\
[outdims,ARR](TensMapT<T>& out, const std::vector<TensMapT<T>>& args){\
	out = args[0].maximum(::eigen::internal::dim_copy<N>(ARR)).reshape(outdims);\
});

/// Return Eigen data object representing reduction where aggregation is max
template <typename T>
EigenptrT reduce_max (teq::Shape outshape, const teq::TensptrT& in, const marsh::iAttributed& attrib)
{
	std::set<teq::RankT> ranks;
	Packer<std::set<teq::RankT>>().unpack(ranks, attrib);

	_MATRIX_REDUCE(maxCoeff);

	teq::RanksT vranks(ranks.begin(), ranks.end());

	DimensionsT outdims = shape_convert(outshape);
	_ARRAY_SWITCH(vranks, _EIGEN_RMAX_CASE)
}

#undef _EIGEN_RMAX_CASE

#undef _MATRIX_REDUCE

/// Return Eigen data object that rgmax in tensor at return_dim
template <typename T>
EigenptrT argmax (teq::Shape outshape, const teq::TensptrT& in, const marsh::iAttributed& attrib)
{
	Packer<teq::RankT> packer;
	bool alldims = attrib.get_attr(packer.get_key()) == nullptr;
	teq::RankT return_dim;
	DimensionsT outdims = shape_convert(outshape);
	if (false == alldims)
	{
		packer.unpack(return_dim, attrib);
		alldims = return_dim >= teq::rank_cap;
	}
	teq::Shape inshape = in->shape();
	if (is_2d(inshape))
	{
		if (alldims || return_dim > 1)
		{
			return internal::unary_matop_smat2mat<T>(outshape, *in,
			[](MatMapT<T>& out, const MatMapT<T>& arg)
			{
				size_t col;
				size_t row;
				arg.maxCoeff(&row, &col);
				*(out.data()) = col + row * arg.cols();
			});
		}
		else if (return_dim == 0)
		{
			return internal::unary_matop_smat2mat<T>(outshape, *in,
			[](MatMapT<T>& out, const MatMapT<T>& arg)
			{
				size_t index;
				T* dst = out.data();
				for (size_t i = 0, nout_rows = out.rows(); i < nout_rows; ++i)
				{
					arg.row(i).maxCoeff(&index);
					dst[i] = index;
				}
			});
		}
		else // if (return_dim == 1)
		{
			return internal::unary_matop_smat2mat<T>(outshape, *in,
			[](MatMapT<T>& out, const MatMapT<T>& arg)
			{
				size_t index;
				T* dst = out.data();
				for (size_t i = 0, nout_cols = out.cols(); i < nout_cols; ++i)
				{
					arg.col(i).maxCoeff(&index);
					dst[i] = index;
				}
			});
		}
	}
	if (alldims)
	{
		return std::make_shared<TensOp<T>>(outshape,teq::CTensT{in.get()},
		[outdims](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
		{
			out = args[0].argmax().template cast<T>().reshape(outdims);
		});
	}
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{in.get()},
	[return_dim,outdims](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].argmax(return_dim).template cast<T>().reshape(outdims);
	});
}

/// Return Eigen data object representing data broadcast across dimensions
template <typename T>
EigenptrT extend (teq::Shape outshape, const teq::TensptrT& in, const marsh::iAttributed& attrib)
{
	auto inshape = in->shape();
	teq::DimsT bcast = *unpack_extend(inshape, attrib);

	if (is_2d(outshape))
	{
		auto incols = inshape.at(0);
		auto inrows = inshape.at(1);
		teq::DimT colext = 1;
		teq::DimT rowext = 1;
		if (bcast.size() > 0)
		{
			colext = bcast.at(0);
		}
		if (bcast.size() > 1)
		{
			rowext = bcast.at(1);
		}
		if (colext > 1 && rowext > 1)
		{
			return internal::unary_smatop<T>(outshape, *in,
			[rowext,colext,incols,inrows](const SMatMapT<T>& arg)
			{
				auto nzs = arg.nonZeros();
				TripletsT<T> trips;
				trips.reserve(rowext * colext * nzs);
				for (int i = 0, n = arg.outerSize(); i < n; ++i)
				{
					for (typename SMatMapT<T>::InnerIterator it(arg, i); it; ++it)
					{
						for (size_t j = 0; j < rowext; ++j)
						{
							for (size_t k = 0; k < colext; ++k)
							{
								trips.emplace_back(it.row() + inrows * j, it.col() + incols * k, it.value());
							}
						}
					}
				}
				SMatrixT<T> out(inrows * rowext, incols * colext);
				out.setFromTriplets(trips.begin(), trips.end());
				return out;
			},
			[rowext,colext,incols,inrows](MatMapT<T>& out, const MatrixT<T>& arg)
			{
				for (size_t i = 0; i < rowext; ++i)
				{
					for (size_t j = 0; j < colext; ++j)
					{
						out.block(i * inrows, j * incols, inrows, incols) = arg;
					}
				}
			});
		}
		else if (colext > 1)
		{
			return internal::unary_smatop<T>(outshape, *in,
			[colext,incols,inrows](const SMatMapT<T>& arg)
			{
				auto nzs = arg.nonZeros();
				TripletsT<T> trips;
				trips.reserve(colext * nzs);
				for (int i = 0, n = arg.outerSize(); i < n; ++i)
				{
					for (typename SMatMapT<T>::InnerIterator it(arg, i); it; ++it)
					{
						for (size_t j = 0; j < colext; ++j)
						{
							trips.emplace_back(it.row(), it.col() + incols * j, it.value());
						}
					}
				}
				SMatrixT<T> out(inrows, incols * colext);
				out.setFromTriplets(trips.begin(), trips.end());
				return out;
			},
			[colext,incols,inrows](MatMapT<T>& out, const MatrixT<T>& arg)
			{
				for (size_t i = 0; i < colext; ++i)
				{
					out.block(0, i * incols, inrows, incols) = arg;
				}
			});
		}
		else if (rowext > 1)
		{
			return internal::unary_smatop<T>(outshape, *in,
			[rowext,incols,inrows](const SMatMapT<T>& arg)
			{
				auto nzs = arg.nonZeros();
				TripletsT<T> trips;
				trips.reserve(rowext * nzs);
				for (int i = 0, n = arg.outerSize(); i < n; ++i)
				{
					for (typename SMatMapT<T>::InnerIterator it(arg, i); it; ++it)
					{
						for (size_t j = 0; j < rowext; ++j)
						{
							trips.emplace_back(it.row() + inrows * j, it.col(), it.value());
						}
					}
				}
				SMatrixT<T> out(inrows * rowext, incols);
				out.setFromTriplets(trips.begin(), trips.end());
				return out;
			},
			[rowext,incols,inrows](MatMapT<T>& out, const MatrixT<T>& arg)
			{
				for (size_t i = 0; i < rowext; ++i)
				{
					out.block(i * inrows, 0, inrows, incols) = arg;
				}
			});
		}
		else
		{
			return std::make_shared<TensRef>(*in);
		}
	}
	teq::ShapeT coord;
	std::fill(coord.begin(), coord.end(), 1);
	std::copy(bcast.begin(), bcast.begin() +
		std::min((size_t) teq::rank_cap, bcast.size()), coord.begin());
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{in.get()},
	[coord](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].broadcast(coord);
	});
}

/// Return Eigen data object representing transpose and permutation
template <typename T>
EigenptrT permute (teq::Shape outshape, const teq::TensptrT& in, const marsh::iAttributed& attrib)
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
	teq::RanksT ranged = arrs::range<teq::RankT>(0,
		teq::narrow_shape(in->shape()).size());
	if (std::equal(ranged.begin(), ranged.end(), order.begin()))
	{
		return std::make_shared<TensRef>(*in);
	}
	auto reorder = internal::dim_copy<teq::rank_cap,teq::RankT>(order);
	if (is_2d(outshape) && is_2d(in->shape()))
	{
		// use matrix when possible
		return internal::unary_smatop<T>(outshape, *in,
		[](const SMatMapT<T>& arg)
		{
			return arg.transpose();
		},
		[](MatMapT<T>& out, const MatMapT<T>& arg)
		{
			out = arg.transpose();
		});
	}
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{in.get()},
	[reorder](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
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
	std::fill(extents.begin(), extents.end(), 1);
	auto shapel = shape.to_list();
	std::copy(shapel.begin(), shapel.end(), extents.begin());
	size_t n = std::min(encoding.size(), (size_t) teq::rank_cap);
	for (size_t i = 0; i < n; ++i)
	{
		teq::DimT offset = std::min(encoding[i].first, (teq::DimT) (shape.at(i) - 1));
		offsets[i] = offset;
		extents[i] = std::min(encoding[i].second, (teq::DimT) (shape.at(i) - offset));
	}
	auto slist = teq::narrow_shape(shape);
	auto sparse_op =
	[offsets,extents](const SMatMapT<T>& arg)
	{
		return arg.
			middleRows(offsets.at(1), extents.at(1)).
			middleCols(offsets.at(0), extents.at(0));
	};
	if (slist.size() > 0 && outshape.compatible_before(
		shape, slist.size() - 1))
	{
		teq::RankT lastdim = slist.size() - 1;
		// only slicing the last dimension
		teq::DimT index = offsets[lastdim];
		teq::NElemT batchsize = shape.n_elems() / shape.at(lastdim);
		// since tensor is column major, index of last dimension denote
		// the number of batches before start of output slice
		// (a batch defined as the subtensor of shape shape[:lastdim])
		auto incr = index * batchsize;
		if (incr == 0)
		{
			return std::make_shared<TensRef>(*in);
		}
		if (is_sparse(*in)) // sparse input does not work with below optimization
		{
			return std::make_shared<SparseMatOp<T>>(teq::CTensT{in.get()},
			[sparse_op](const teq::CTensT& args)
			{
				auto smat = make_smatmap<T>(*args.front());
				return sparse_op(smat.get());
			});
		}
		return std::make_shared<UnsafeTensRef<T>>(*in, index * batchsize);
	}
	DimensionsT outdims = shape_convert(outshape);
	if (is_2d(shape))
	{
		// use matrix when possible
		return internal::unary_smatop<T>(outshape, *in, sparse_op,
		[offsets,extents](MatMapT<T>& out, const MatMapT<T>& arg)
		{
			out = arg.block(
				offsets.at(1), offsets.at(0),
				extents.at(1), extents.at(0));
		});
	}
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{in.get()},
	[offsets,extents,outdims](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
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
	if (is_2d(outshape))
	{
		// use matrix when possible
		return internal::unary_matop_smat2mat<T>(outshape, in,
		[paddings](MatMapT<T>& out, const MatMapT<T>& arg)
		{
			out.setZero();
			out.block(paddings.at(1).first, paddings.at(0).first, arg.rows(), arg.cols()) = arg;
		});
	}
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{&in},
	[paddings](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
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
	if (is_2d(in.shape()))
	{
		using StrideT = Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>;
		// use matrix when possible
		return internal::unary_matop_smat2mat<T>(outshape, in,
		[incrs](MatMapT<T>& out, const MatMapT<T>& arg)
		{
			auto ncols = out.cols();
			out = Eigen::Map<MatrixT<T>,Eigen::Unaligned,StrideT>(
				(T*) arg.data(), out.rows(), ncols,
				StrideT(incrs[1] * ncols, incrs[0]));
		});
	}
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{&in},
	[incrs](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
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
	if (is_2d(outshape))
	{
		using StrideT = Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>;
		// use matrix when possible
		return internal::unary_matop_smat2mat<T>(outshape, in,
		[incrs](MatMapT<T>& out, const MatMapT<T>& arg)
		{
			Eigen::Map<MatrixT<T>,Eigen::Unaligned,StrideT>(
				out.data(), arg.rows(), arg.cols(),
				StrideT(incrs[1] * out.cols(), incrs[0])) = arg;
		});
	}
	return std::make_shared<TensOp<T>>(outshape, teq::CTensT{&in},
	[incrs](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out.setZero();
		out.stride(incrs) = args[0];
	});
}

template <typename T>
EigenptrT reverse (teq::Shape outshape, const teq::TensptrT& in, const marsh::iAttributed& attrib)
{
	std::set<teq::RankT> dims;
	Packer<std::set<teq::RankT>>().unpack(dims, attrib);

	std::array<bool,teq::rank_cap> do_reverse;
	std::fill(do_reverse.begin(), do_reverse.end(), false);
	for (teq::RankT i : dims)
	{
		do_reverse[i] = true;
	}
	if (is_2d(outshape))
	{
		// use matrix when possible
		if (do_reverse[0] && do_reverse[1])
		{
			return internal::unary_matop_smat2mat<T>(outshape, *in,
			[](MatMapT<T>& out, const MatMapT<T>& arg)
			{
				out = arg.reverse();
			});
		}
		else if (do_reverse[0])
		{
			return internal::unary_matop_smat2mat<T>(outshape, *in,
			[](MatMapT<T>& out, const MatMapT<T>& arg)
			{
				out = arg.rowwise().reverse();
			});
		}
		else if (do_reverse[1])
		{
			return internal::unary_matop_smat2mat<T>(outshape, *in,
			[](MatMapT<T>& out, const MatMapT<T>& arg)
			{
				out = arg.colwise().reverse();
			});
		}
		else
		{
			return std::make_shared<TensRef>(*in);
		}
	}
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{in.get()},
	[do_reverse](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].reverse(do_reverse);
	});
}

template <typename T>
EigenptrT concat (teq::Shape outshape, const teq::TensptrsT& group, const marsh::iAttributed& attrib)
{
	assert(group.size() > 1);
	teq::CTensT args;
	args.reserve(group.size());
	std::transform(group.begin(), group.end(), std::back_inserter(args),
	[](teq::TensptrT arg)
	{
		return arg.get();
	});
	teq::RankT axis;
	Packer<teq::RankT>().unpack(axis, attrib);
	if (is_2d(outshape))
	{
		if (axis == 0)
		{
			if (std::all_of(args.begin(), args.end(),
			[](const teq::iTensor* arg)
			{
				return is_sparse(*arg);
			}))
			{
				return std::make_shared<SparseMatOp<T>>(args,
				[outshape](const teq::CTensT& args)
				{
					TripletsT<T> trips;
					teq::DimT cols = 0;
					for (auto& arg : args)
					{
						auto smat = make_smatmap<T>(*arg);
						auto nzs = smat->nonZeros();
						for (int i = 0, n = smat->outerSize(); i < n; ++i)
						{
							for (typename SMatMapT<T>::InnerIterator it(smat.get(), i); it; ++it)
							{
								trips.emplace_back(it.row(), it.col() + cols, it.value());
							}
						}
						cols += smat->cols();
					}
					SMatrixT<T> out(outshape.at(1), outshape.at(0));
					out.setFromTriplets(trips.begin(), trips.end());
					return out;
				});
			}
			return internal::nnary_matop<T>(outshape, args,
			[](MatMapT<T>& out, const teq::iTensor& arg)
			{
				auto inshape = arg.shape();
				auto nrows = inshape.at(1);
				auto ncols = inshape.at(0);
				if (is_sparse(arg))
				{
					out.block(0, 0, nrows, ncols) = make_smatmap<T>(arg).get();
				}
				else
				{
					out.block(0, 0, nrows, ncols) = make_matmap<T>(arg).get();
				}
				return ncols;
			},
			[](MatMapT<T>& out, const teq::iTensor& arg, size_t iter)
			{
				auto inshape = arg.shape();
				auto ncols = inshape.at(0);
				out.block(0, iter, inshape.at(1), ncols) = make_smatmap<T>(arg).get();
				return iter + ncols;
			},
			[](MatMapT<T>& out, const teq::iTensor& arg, size_t iter)
			{
				auto inshape = arg.shape();
				auto ncols = inshape.at(0);
				out.block(0, iter, inshape.at(1), ncols) = make_matmap<T>(arg).get();
				return iter + ncols;
			});
		}
		else // if (axis == 1) otherwise outshape can't be 2d...
		 {
			if (std::all_of(args.begin(), args.end(),
			[](const teq::iTensor* arg)
			{
				return is_sparse(*arg);
			}))
			{
				return std::make_shared<SparseMatOp<T>>(args,
				[outshape](const teq::CTensT& args)
				{
					TripletsT<T> trips;
					teq::DimT rows = 0;
					for (auto& arg : args)
					{
						auto smat = make_smatmap<T>(*arg);
						auto nzs = smat->nonZeros();
						for (int i = 0, n = smat->outerSize(); i < n; ++i)
						{
							for (typename SMatMapT<T>::InnerIterator it(smat.get(), i); it; ++it)
							{
								trips.emplace_back(it.row() + rows, it.col(), it.value());
							}
						}
						rows += smat->rows();
					}
					SMatrixT<T> out(outshape.at(1), outshape.at(0));
					out.setFromTriplets(trips.begin(), trips.end());
					return out;
				});
			}
			return internal::nnary_matop<T>(outshape, args,
			[](MatMapT<T>& out, const teq::iTensor& arg)
			{
				auto inshape = arg.shape();
				auto nrows = inshape.at(1);
				auto ncols = inshape.at(0);
				if (is_sparse(arg))
				{
					out.block(0, 0, nrows, ncols) = make_smatmap<T>(arg).get();
				}
				else
				{
					out.block(0, 0, nrows, ncols) = make_matmap<T>(arg).get();
				}
				return nrows;
			},
			[](MatMapT<T>& out, const teq::iTensor& arg, size_t iter)
			{
				auto inshape = arg.shape();
				auto nrows = inshape.at(1);
				out.block(iter, 0, nrows, inshape.at(0)) = make_smatmap<T>(arg).get();
				return iter + nrows;
			},
			[](MatMapT<T>& out, const teq::iTensor& arg, size_t iter)
			{
				auto inshape = arg.shape();
				auto nrows = inshape.at(1);
				out.block(iter, 0, nrows, inshape.at(0)) = make_matmap<T>(arg).get();
				return iter + nrows;
			});
		}
	}
	if (group.size() == 2)
	{
		return std::make_shared<TensOp<T>>(outshape, args,
		[axis](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
		{
			out = args[0].concatenate(args[1],axis);
		});
	}
	std::array<Eigen::Index,teq::rank_cap-1> reshaped;
	auto outlist = outshape.to_list();
	auto it = outlist.begin();
	std::copy(it, it + axis, reshaped.begin());
	std::copy(it + axis + 1, outlist.end(), reshaped.begin() + axis);
	return std::make_shared<TensOp<T>>(outshape, args,
	[axis,reshaped](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
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
		return internal::unary_matop<T>(outshape, in,
		[](MatMapT<T>& out, const teq::CTensT& args)
		{
			out = make_smatmap<T>(*args.front())->cwiseAbs();
		},
		[](MatMapT<T>& out, const MatMapT<T>& arg)
		{
			out = arg.cwiseAbs();
		});
	}
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{&in},
	[](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
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
		return internal::unary_matop<T>(outshape, in,
		[](MatMapT<T>& out, const teq::CTensT& args)
		{
			out = -make_smatmap<T>(*args.front()).get();
		},
		[](MatMapT<T>& out, const MatMapT<T>& arg)
		{
			out = -arg;
		});
	}
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{&in},
	[](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = -args[0];
	});
}

/// Given reference to output array, and input vector ref,
/// make output elements take sine of inputs
template <typename T>
EigenptrT sin (teq::Shape outshape, const teq::iTensor& in)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		return internal::unary_matop_smat2mat<T>(outshape, in,
		[](MatMapT<T>& out, const MatMapT<T>& arg)
		{
			out = arg.array().sin();
		});
	}
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{&in},
	[](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
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
	if (is_2d(outshape))
	{
		// use matrix when possible
		return internal::unary_matop_smat2mat<T>(outshape, in,
		[](MatMapT<T>& out, const MatMapT<T>& arg)
		{
			out = arg.array().cos();
		});
	}
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{&in},
	[](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
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
		return internal::unary_matop_smat2mat<T>(outshape, in,
		[](MatMapT<T>& out, const MatMapT<T>& arg)
		{
			out = arg.array().tan();
		});
	}
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{&in},
	[](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
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
		return internal::unary_matop_smat2mat<T>(outshape, in,
		[](MatMapT<T>& out, const MatMapT<T>& arg)
		{
			out = arg.array().exp();
		});
	}
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{&in},
	[](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
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
		return internal::unary_matop_smat2mat<T>(outshape, in,
		[](MatMapT<T>& out, const MatMapT<T>& arg)
		{
			out = arg.array().log();
		});
	}
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{&in},
	[](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
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
		return internal::unary_matop<T>(outshape, in,
		[](MatMapT<T>& out, const teq::CTensT& args)
		{
			out = make_smatmap<T>(*args.front())->cwiseSqrt();
		},
		[](MatMapT<T>& out, const MatMapT<T>& arg)
		{
			out = arg.cwiseSqrt();
		});
	}
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{&in},
	[](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
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
		return internal::unary_matop_smat2mat<T>(outshape, in,
		[](MatMapT<T>& out, const MatMapT<T>& arg)
		{
			out = arg.array().round();
		});
	}
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{&in},
	[](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].round();
	});
}

template <typename T>
EigenptrT sigmoid (teq::Shape outshape, const teq::iTensor& in)
{
	if (is_2d(outshape))
	{
		return internal::unary_matop<T>(outshape, in,
		[](MatMapT<T>& out, const teq::CTensT& args)
		{
			out = make_matmap<T>(*args.front())->unaryExpr(Eigen::internal::scalar_sigmoid_op<T>());
		},
		[](MatMapT<T>& out, const MatMapT<T>& arg)
		{
			out = arg.unaryExpr(Eigen::internal::scalar_sigmoid_op<T>());
		});
	}
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{&in},
	[](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
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
		return internal::unary_matop<T>(outshape, in,
		[](MatMapT<T>& out, const teq::CTensT& args)
		{
			out = make_matmap<T>(*args.front())->unaryExpr(Eigen::internal::scalar_tanh_op<T>());
		},
		[](MatMapT<T>& out, const MatMapT<T>& arg)
		{
			out = arg.unaryExpr(Eigen::internal::scalar_tanh_op<T>());
		});
	}
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{&in},
	[](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
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
		return internal::unary_matop<T>(outshape, in,
		[](MatMapT<T>& out, const teq::CTensT& args)
		{
			out = make_matmap<T>(*args.front())->unaryExpr(Eigen::internal::scalar_square_op<T>());
		},
		[](MatMapT<T>& out, const MatMapT<T>& arg)
		{
			out = arg.unaryExpr(Eigen::internal::scalar_square_op<T>());
		});
	}
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{&in},
	[](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
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
		return internal::unary_matop<T>(outshape, in,
		[](MatMapT<T>& out, const teq::CTensT& args)
		{
			out = make_matmap<T>(*args.front())->unaryExpr(Eigen::internal::scalar_cube_op<T>());
		},
		[](MatMapT<T>& out, const MatMapT<T>& arg)
		{
			out = arg.unaryExpr(Eigen::internal::scalar_cube_op<T>());
		});
	}
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{&in},
	[](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
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
		return internal::binaryexpr_matop<T>(outshape, a, b,
		[](const T& a, const T& b) -> T
		{
			return std::pow(a, b);
		});
	}
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{&a,&b},
	[](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].binaryExpr(args[1],
		BinaryF<T>(
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
	teq::CTensT args;
	args.reserve(group.size());
	std::transform(group.begin(), group.end(), std::back_inserter(args),
	[](teq::TensptrT arg)
	{
		return arg.get();
	});
	if (is_2d(outshape))
	{
		// use matrix when possible
		return internal::nnary_matop<T>(outshape, args,
		[](MatMapT<T>& out, const teq::iTensor& arg)
		{
			if (is_sparse(arg))
			{
				out = make_smatmap<T>(arg).get();
			}
			else
			{
				out = make_matmap<T>(arg).get();
			}
			return 0;
		},
		[](MatMapT<T>& out, const teq::iTensor& arg, size_t iter)
		{
			out += make_smatmap<T>(arg).get();
			return 0;
		},
		[](MatMapT<T>& out, const teq::iTensor& arg, size_t iter)
		{
			out += make_matmap<T>(arg).get();
			return 0;
		});
	}
	if (group.size() == 2)
	{
		const teq::TensptrT& a = group[0];
		const teq::TensptrT& b = group[1];
		return std::make_shared<TensOp<T>>(outshape,teq::CTensT{a.get(),b.get()},
		[](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
		{
			out = args[0] + args[1];
		});
	}
	return std::make_shared<TensOp<T>>(outshape,args,
	[](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
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
		return std::make_shared<GenericMatOp<T>>(outshape,teq::CTensT{&a,&b},
		[](MatMapT<T>& out, const teq::CTensT& args)
		{
			auto& a = *args.front();
			auto& b = *args.back();
			auto asparse = is_sparse(a);
			auto bsparse = is_sparse(b);
			if (asparse && bsparse)
			{
				out = make_smatmap<T>(a).get() - make_smatmap<T>(b).get();
			}
			else if (asparse && !bsparse)
			{
				out = make_smatmap<T>(a).get() - make_matmap<T>(b).get();
			}
			else if (!asparse && bsparse)
			{
				out = make_matmap<T>(a).get() - make_smatmap<T>(b).get();
			}
			else
			{
				out = make_matmap<T>(a).get() - make_matmap<T>(b).get();
			}
		});
	}
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{&a,&b},
	[](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0] - args[1];
	});
}

template <typename T>
EigenptrT mul (teq::Shape outshape, const teq::TensptrsT& group)
{
	assert(group.size() > 1);
	teq::CTensT args;
	args.reserve(group.size());
	std::transform(group.begin(), group.end(), std::back_inserter(args),
	[](teq::TensptrT arg)
	{
		return arg.get();
	});
	if (is_2d(outshape))
	{
		if (std::any_of(args.begin(), args.end(),
		[](const teq::iTensor* arg)
		{
			return is_sparse(*arg);
		}))
		{
			return std::make_shared<SparseMatOp<T>>(args,
			[](const teq::CTensT& args) -> SMatrixT<T>
			{
				SMatrixT<T> out;
				if (is_sparse(*args.front()))
				{
					out = make_smatmap<T>(*args.front()).get();
				}
				else
				{
					out = make_matmap<T>(*args.front())->sparseView();
				}
				for (size_t i = 1, n = args.size(); i < n; ++i)
				{
					auto& arg = args[i];
					if (is_sparse(*arg))
					{
						out = make_smatmap<T>(*arg)->cwiseProduct(out);
					}
					else
					{
						out = out.cwiseProduct(make_matmap<T>(*arg).get());
					}
				}
				return out;
			});
		}
		return std::make_shared<MatOp<T>>(outshape, args,
		[](MatMapT<T>& out, const std::vector<MatMapT<T>>& args)
		{
			out = args.front();
			for (size_t i = 1, n = args.size(); i < n; ++i)
			{
				out = out.cwiseProduct(args[i]);
			}
		});
	}
	if (group.size() == 2)
	{
		const teq::TensptrT& a = group[0];
		const teq::TensptrT& b = group[1];
		return std::make_shared<TensOp<T>>(outshape,teq::CTensT{a.get(),b.get()},
		[](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
		{
			out = args[0] * args[1];
		});
	}
	return std::make_shared<TensOp<T>>(outshape,args,
	[outshape](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0];
		for (size_t i = 1, n = args.size(); i < n; ++i)
		{
			out *= args[i];
		}
	});
}

#define _BINARY_MATOP_CRISSCROSS(OP)\
	if (is_2d(outshape)){\
		auto asparse = is_sparse(a);\
		auto bsparse = is_sparse(b);\
		if (asparse && bsparse){\
			return std::make_shared<SparseMatOp<T>>(teq::CTensT{&a,&b},\
			[](const teq::CTensT& args) -> SMatrixT<T>{\
				return make_smatmap<T>(*args.front())->OP(make_smatmap<T>(*args.back()).get());\
			});\
		}else if (asparse && !bsparse){\
			return std::make_shared<GenericMatOp<T>>(outshape,teq::CTensT{&a,&b},\
			[](MatMapT<T>& out, const teq::CTensT& args){\
				MatrixT<T> am = make_smatmap<T>(*args.front()).get();\
				out = am.OP(make_matmap<T>(*args.back()).get());\
			});\
		}else if (!asparse && bsparse){\
			return std::make_shared<GenericMatOp<T>>(outshape,teq::CTensT{&a,&b},\
			[](MatMapT<T>& out, const teq::CTensT& args){\
				MatrixT<T> bm = make_smatmap<T>(*args.back()).get();\
				out = make_matmap<T>(*args.front())->OP(bm);\
			});\
		}else{\
			return std::make_shared<MatOp<T>>(outshape, teq::CTensT{&a,&b},\
			[](MatMapT<T>& out, const std::vector<MatMapT<T>>& args){\
				out = args.front().OP(args.back());\
			});\
		}\
	}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index divide
/// Only accept 2 arguments
template <typename T>
EigenptrT div (teq::Shape outshape, const teq::iTensor& a, const teq::iTensor& b)
{
	if (is_sparse(b))
	{
		global::warn("denominator is sparse, most likely divide by zero");
	}
	_BINARY_MATOP_CRISSCROSS(cwiseQuotient)
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{&a,&b},
	[](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
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
		return internal::binaryexpr_matop<T>(outshape, a, b,
		[](const T& a, const T& b) -> T
		{
			return a == b;
		});
	}
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{&a,&b},
	[](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].binaryExpr(args[1],
		BinaryF<T>(
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
		return internal::binaryexpr_matop<T>(outshape, a, b,
		[](const T& a, const T& b) -> T
		{
			return a != b;
		});
	}
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{&a,&b},
	[](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].binaryExpr(args[1],
		BinaryF<T>(
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
		return internal::binaryexpr_matop<T>(outshape, a, b,
		[](const T& a, const T& b) -> T
		{
			return a < b;
		});
	}
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{&a,&b},
	[](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].binaryExpr(args[1],
		BinaryF<T>(
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
		return internal::binaryexpr_matop<T>(outshape, a, b,
		[](const T& a, const T& b) -> T
		{
			return a > b;
		});
	}
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{&a,&b},
	[](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].binaryExpr(args[1],
		BinaryF<T>(
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
	_BINARY_MATOP_CRISSCROSS(cwiseMin);
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{&a,&b},
	[](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].cwiseMin(args[1]);
	});
}

/// Given arguments, for every mapped index i in range [0:max_nelems],
/// take the maximum all elements for all arguments
template <typename T>
EigenptrT max (teq::Shape outshape, const teq::iTensor& a, const teq::iTensor& b)
{
	_BINARY_MATOP_CRISSCROSS(cwiseMax)
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{&a,&b},
	[](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].cwiseMax(args[1]);
	});
}

#undef _BINARY_MATOP_CRISSCROSS

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply std::uniform_distributon function
/// Only accept 2 arguments
template <typename T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
EigenptrT rand_uniform (teq::Shape outshape, const teq::iTensor& a, const teq::iTensor& b)
{
	if (is_2d(outshape))
	{
		// use matrix when possible
		auto generator = global::get_generator();
		return internal::binaryexpr_matop<T>(outshape, a, b,
		[generator](const T& a, const T& b) -> T
		{
			return generator->unif_int(a, b);
		});
	}
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{&a,&b},
	[](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		auto generator = global::get_generator();
		out = args[0].binaryExpr(args[1],
		BinaryF<T>(
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
		auto generator = global::get_generator();
		return internal::binaryexpr_matop<T>(outshape, a, b,
		[generator](const T& a, const T& b) -> T
		{
			return generator->unif_dec(a, b);
		});
	}
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{&a,&b},
	[](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		auto generator = global::get_generator();
		out = args[0].binaryExpr(args[1],
		BinaryF<T>(
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
		return std::make_shared<GenericMatOp<T>>(outshape,teq::CTensT{&condition,&then,&otherwise},
		[](MatMapT<T>& out, const teq::CTensT& args)
		{
			auto& cond = *args.front();
			auto& a = *args[1];
			auto& b = *args[2];
			MatrixT<T> condm;
			MatrixT<T> am;
			MatrixT<T> bm;
			if (is_sparse(cond))
			{
				condm = make_smatmap<T>(cond).get();
			}
			else
			{
				condm = make_matmap<T>(cond).get();
			}
			if (is_sparse(a))
			{
				am = make_smatmap<T>(a).get();
			}
			else
			{
				am = make_matmap<T>(a).get();
			}
			if (is_sparse(b))
			{
				bm = make_smatmap<T>(b).get();
			}
			else
			{
				bm = make_matmap<T>(b).get();
			}
			out = condm.select(am, bm);
		});
	}
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{&condition,&then,&otherwise},
	[](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].select(args[1],args[2]);
	});
}

#define _EIGEN_CONTRACT_CASE(ARR, N)\
return std::make_shared<TensOp<T>>(outshape,teq::CTensT{&a,&b},\
[ARR,outdims](TensMapT<T>& out, const std::vector<TensMapT<T>>& args){\
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
		auto asparse = is_sparse(a);
		auto bsparse = is_sparse(b);
		if (asparse && bsparse)
		{
			return std::make_shared<SparseMatOp<T>>(teq::CTensT{&a,&b},
			[](const teq::CTensT& args) -> SMatrixT<T>
			{
				return make_smatmap<T>(*args.front()).get() * make_smatmap<T>(*args.back()).get();
			});
		}
		else if (asparse && !bsparse)
		{
			return std::make_shared<GenericMatOp<T>>(outshape,teq::CTensT{&a,&b},
			[](MatMapT<T>& out, const teq::CTensT& args)
			{
			auto a = make_smatmap<T>(*args.front()).get();
			auto b = make_matmap<T>(*args.back()).get();
				out = a * b;
			});
		}
		else if (!asparse && bsparse)
		{
			return std::make_shared<GenericMatOp<T>>(outshape,teq::CTensT{&a,&b},
			[](MatMapT<T>& out, const teq::CTensT& args)
			{
				out = make_matmap<T>(*args.front()).get() * make_smatmap<T>(*args.back()).get();
			});
		}
		else
		{
			return std::make_shared<MatOp<T>>(outshape,teq::CTensT{&a,&b},
			[](MatMapT<T>& out, const std::vector<MatMapT<T>>& args)
			{
				out = args.front() * args.back();
			});
		}
	}
	DimensionsT outdims = shape_convert(outshape);
	_ARRAY_SWITCH(dims, _EIGEN_CONTRACT_CASE);
}

#undef _EIGEN_CONTRACT_CASE

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
		return std::make_shared<TensOp<T>>(outshape,teq::CTensT{&a,&b},
		[nbatches,os,as,bs](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
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
	PairVecT<teq::RankT> dims = {{0, 1}};
	Packer<PairVecT<teq::RankT>>().pack(contract_attr, dims);
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

	if (is_2d(outshape) && is_2d(input.shape()) && is_2d(kshape))
	{
		auto isparse = is_sparse(input);
		auto ksparse = is_sparse(kernel);
		if (isparse && ksparse)
		{
			return std::make_shared<GenericMatOp<T>>(outshape, teq::CTensT{&input, &kernel},
			[](MatMapT<T>& out, const teq::CTensT& args)
			{
				auto img = make_smatmap<T>(*args.front());
				auto krn = make_smatmap<T>(*args.back());
				auto kr = krn->rows();
				auto kc = krn->cols();
				for (size_t r = 0, nr = img->rows() - kr; r < nr; ++r)
				{
					for (size_t c = 0, nc = img->cols() - kc; c < nc; ++c)
					{
						out(r, c) = (img->middleRows(r, kr).middleCols(c, kc) * krn.get()).sum();
					}
				}
			});
		}
		else if (isparse)
		{
			return std::make_shared<GenericMatOp<T>>(outshape, teq::CTensT{&input, &kernel},
			[](MatMapT<T>& out, const teq::CTensT& args)
			{
				auto img = make_smatmap<T>(*args.front());
				auto krn = make_matmap<T>(*args.back());
				auto kr = krn->rows();
				auto kc = krn->cols();
				for (size_t r = 0, nr = img->rows() - kr; r < nr; ++r)
				{
					for (size_t c = 0, nc = img->cols() - kc; c < nc; ++c)
					{
						out(r, c) = (img->middleRows(r, kr).middleCols(c, kc) * krn.get()).sum();
					}
				}
			});
		}
		else if (ksparse)
		{
			return std::make_shared<GenericMatOp<T>>(outshape, teq::CTensT{&input, &kernel},
			[](MatMapT<T>& out, const teq::CTensT& args)
			{
				auto img = make_matmap<T>(*args.front());
				auto krn = make_smatmap<T>(*args.back());
				auto kr = krn->rows();
				auto kc = krn->cols();
				for (size_t r = 0, nr = img->rows() - kr; r < nr; ++r)
				{
					for (size_t c = 0, nc = img->cols() - kc; c < nc; ++c)
					{
						out(r, c) = (img->block(r, c, kr, c) * krn.get()).sum();
					}
				}
			});
		}
	}

	teq::ShapeT dims;
	std::copy(order.begin(), order.end(), dims.begin());
	return std::make_shared<TensOp<T>>(outshape,teq::CTensT{&input,&kernel},
	[dims](TensMapT<T>& out, const std::vector<TensMapT<T>>& args)
	{
		out = args[0].convolve(args[1],dims);
	});
}

#define ASSIGN_OP(ASS)\
if (is_2d(target.shape())){\
	if (is_sparse(target)){\
		return std::make_shared<SparseMatAssign<T>>(target, source,\
		[](SMatMapT<T>& target, const SMatMapT<T>& source){\
			target ASS source;\
		});\
	}return std::make_shared<MatAssign<T>>(target, source,\
	[](MatMapT<T>& target, const teq::iTensor& source){\
		if (is_sparse(source)){\
			target ASS make_smatmap<T>(source).get();\
		}else{\
			target ASS make_matmap<T>(source).get();\
		}\
	});\
}return std::make_shared<TensAssign<T>>(target, source,\
[](TensMapT<T>& target, const TensMapT<T>& source){\
	target ASS source;\
});

template <typename T>
EigenptrT assign (teq::iTensor& target, const teq::iTensor& source)
{
	ASSIGN_OP(=)
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

#define _EIGEN_MATCAST_CASE(INTYPE)\
if (is_sparse(*input)){\
	out = std::make_shared<SparseMatOp<T>>(teq::CTensT{input.get()},\
	[](const teq::CTensT& args) -> SMatrixT<T> {\
		return make_smatmap<INTYPE>(*args.front())->template cast<T>();\
	});\
}else{\
	out = std::make_shared<MatOp<T,INTYPE>>(input->shape(), teq::CTensT{input.get()},\
	[](MatMapT<T>& out, const std::vector<MatMapT<INTYPE>>& args){\
		out = args.front().template cast<T>();\
	});\
}

#define _EIGEN_TENSCAST_CASE(INTYPE)\
out = std::make_shared<TensOp<T,INTYPE>>(input->shape(),teq::CTensT{input.get()},\
[](TensMapT<T>& out, const std::vector<TensMapT<INTYPE>>& args){\
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
	if (is_2d(input->shape()))
	{
		TYPE_LOOKUP(_EIGEN_MATCAST_CASE, intype);
	}
	else
	{
		TYPE_LOOKUP(_EIGEN_TENSCAST_CASE, intype);
	}
	return out;
}

#undef _EIGEN_MATCAST_CASE

#undef _EIGEN_TENSCAST_CASE

}


#endif // EIGEN_OPERATOR_HPP
#endif // PERM_OP
