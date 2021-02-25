///
/// eigen.hpp
/// eigen
///
/// Purpose:
/// Define Eigen tensor and matrix transformation functions
///

#ifndef EIGEN_CONVERT_HPP
#define EIGEN_CONVERT_HPP

#include "Eigen/Core"
#include "Eigen/Sparse"
#include "unsupported/Eigen/CXX11/Tensor"

#include "internal/teq/teq.hpp"

namespace eigen
{

/// Eigen shape
using DimensionsT = std::array<Eigen::Index,teq::rank_cap>;

using StorageIdxT = int32_t;

template <typename T>
using TripletT = Eigen::Triplet<T>;

template <typename T>
using TripletsT = std::vector<TripletT<T>>;

/// Sparse Eigen Matrix
template <typename T>
using SMatrixT = Eigen::SparseMatrix<T,Eigen::RowMajor,StorageIdxT>;

/// Eigen Sparse Matrix Map
template <typename T>
using SMatMapT = Eigen::Map<SMatrixT<T>>;

template <typename T>
using SparseBaseT = Eigen::MatrixBase<SMatrixT<T>>; // either SMatrixT or SMatMapT

/// Eigen Matrix
template <typename T>
using MatrixT = Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;

/// Eigen Matrix Map
template <typename T>
using MatMapT = Eigen::Map<MatrixT<T>>;

template <typename T>
using MatBaseT = Eigen::MatrixBase<MatrixT<T>>; // either MatrixT or MatMapT

/// Eigen Tensor
template <typename T>
using TensorT = Eigen::Tensor<T,teq::rank_cap>;

/// Eigen Tensor Map
template <typename T>
using TensMapT = Eigen::TensorMap<TensorT<T>>;

struct SparseInfo final
{
	template <typename T>
	static SparseInfo get (SMatrixT<T>& base)
	{
		return SparseInfo(base.innerIndexPtr(), base.outerIndexPtr(), base.nonZeros());
	}

	SparseInfo (StorageIdxT* inner, StorageIdxT* outer, int64_t non_zeros) :
		inner_indices_(inner), outer_indices_(outer), non_zeros_(non_zeros) {}

	StorageIdxT* inner_indices_;

	StorageIdxT* outer_indices_;

	int64_t non_zeros_;
};

/// Return Matrix Map given Tensor
template <typename T>
inline MatMapT<T> tens_to_matmap (TensorT<T>& tens)
{
	return MatMapT<T>(tens.data(),
		tens.dimension(1), tens.dimension(0));
}

/// Return Map of Matrix
template <typename T>
inline MatMapT<T> mat_to_matmap (MatrixT<T>& mat)
{
	return MatMapT<T>(mat.data(), mat.rows(), mat.cols());
}

/// Return Matrix Map of Tensor Map
template <typename T>
inline MatMapT<T> tensmap_to_matmap (TensMapT<T>& tens)
{
	return MatMapT<T>(tens.data(),
		tens.dimension(1), tens.dimension(0));
}

/// Return Tensor Map of Matrix
template <typename T>
inline TensMapT<T> mat_to_tensmap (MatrixT<T>& mat)
{
	return TensMapT<T>(mat.data(),
		mat.cols(), mat.rows(), 1,1,1,1,1,1);
}

/// Return Tensor Map of Tensor
template <typename T>
inline TensMapT<T> tens_to_tensmap (TensorT<T>& tens)
{
	return TensMapT<T>(tens.data(), tens.dimensions());
}

/// Return Eigen Matrix given raw data and teq Shape
template <typename T>
inline MatMapT<T> make_matmap (T* data, const teq::Shape& shape)
{
	if (nullptr == data)
	{
		global::fatal("cannot get matrix map with null data");
	}
	return MatMapT<T>(data, shape.at(1), shape.at(0));
}

template <typename T>
inline SMatMapT<T> make_smatmap (T* nzdata,
	const SparseInfo& sinfo, const teq::Shape& shape)
{
	if (nullptr == nzdata)
	{
		global::fatal("cannot get sparse matrix map with null data");
	}
	return SMatMapT<T>(shape.at(1), shape.at(0), sinfo.non_zeros_,
		sinfo.outer_indices_, sinfo.inner_indices_, nzdata);
}

/// Return Eigen Tensor given raw data and teq Shape
template <typename T>
inline TensMapT<T> make_tensmap (T* data, const teq::Shape& shape)
{
	auto shapel = shape.to_list();
	std::array<Eigen::Index,teq::rank_cap> slist;
	std::copy(shapel.begin(), shapel.end(), slist.begin());
	if (nullptr == data)
	{
		global::fatal("cannot get tensor map with null data");
	}
	return TensMapT<T>(data, slist);
}

/// Return the teq Shape representation of Eigen Tensor
template <typename T>
teq::Shape get_shape (const TensorT<T>& tens)
{
	auto slist = tens.dimensions();
	return teq::Shape(teq::DimsT(slist.begin(), slist.end()));
}

/// Return the teq Shape representation of Eigen Tensor Map
template <typename T>
teq::Shape get_shape (const TensMapT<T>& tens)
{
	auto slist = tens.dimensions();
	return teq::Shape(teq::DimsT(slist.begin(), slist.end()));
}

/// Return Eigen shape of teq Shape
DimensionsT shape_convert (teq::Shape shape);

}

#endif // EIGEN_CONVERT_HPP
