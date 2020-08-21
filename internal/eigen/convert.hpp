///
/// eigen.hpp
/// eigen
///
/// Purpose:
/// Define Eigen tensor and matrix transformation functions
///

#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"

#include "internal/teq/teq.hpp"

#ifndef EIGEN_CONVERT_HPP
#define EIGEN_CONVERT_HPP

namespace eigen
{

/// Eigen shape
using DimensionsT = std::array<Eigen::Index,teq::rank_cap>;

/// Eigen Matrix
template <typename T>
using  MatrixT = Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;

/// Eigen Matrix Map (reference)
template <typename T>
using MatMapT = Eigen::Map<MatrixT<T>>;

/// Eigen Tensor
template <typename T>
using TensorT = Eigen::Tensor<T,8>;

/// Eigen Tensor Map (reference)
template <typename T>
using TensMapT = Eigen::TensorMap<TensorT<T>>;

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

/// Return Eigen Tensor filled with 0s given teq Shape
template <typename T>
inline TensorT<T> make_tensor (const teq::Shape& shape)
{
	std::array<Eigen::Index,teq::rank_cap> slist;
	std::copy(shape.begin(), shape.end(), slist.begin());
	TensorT<T> out(slist);
	out.setZero();
	return out;
}

/// Return Eigen Matrix given raw data and teq Shape
template <typename T>
inline MatMapT<T> make_matmap (T* data, const teq::Shape& shape)
{
	if (nullptr == data)
	{
		global::fatal("cannot get matmap from nullptr");
	}
	return MatMapT<T>(data, shape.at(1), shape.at(0));
}

/// Return Eigen Tensor given raw data and teq Shape
template <typename T>
inline TensMapT<T> make_tensmap (T* data, const teq::Shape& shape)
{
	std::array<Eigen::Index,teq::rank_cap> slist;
	std::copy(shape.begin(), shape.end(), slist.begin());
	if (nullptr == data)
	{
		global::fatal("cannot get tensmap from nullptr");
	}
	return TensMapT<T>(data, slist);
}

/// Return the teq Shape representation of Eigen Tensor
template <typename T>
teq::Shape get_shape (const TensorT<T>& tens)
{
	auto slist = tens.dimensions();
	return teq::Shape(std::vector<teq::DimT>(slist.begin(), slist.end()));
}

/// Return the teq Shape representation of Eigen Tensor Map
template <typename T>
teq::Shape get_shape (const TensMapT<T>& tens)
{
	auto slist = tens.dimensions();
	return teq::Shape(std::vector<teq::DimT>(slist.begin(), slist.end()));
}

/// Return Eigen shape of teq Shape
DimensionsT shape_convert (teq::Shape shape);

}

#endif // EIGEN_CONVERT_HPP
