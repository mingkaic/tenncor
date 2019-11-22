///
/// eigen.hpp
/// eigen
///
/// Purpose:
/// Define Eigen tensor and matrix transformation functions
///

#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"

#include "teq/shape.hpp"

#ifndef EIGEN_EIGEN_HPP
#define EIGEN_EIGEN_HPP

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

/// Interface of generic Eigen data wrapper
template <typename T>
struct iEigen
{
	virtual ~iEigen (void) = default;

	/// Apply the assignment
	virtual void assign (void) = 0;

	/// Return Eigen object output
	virtual T* get_ptr (void) = 0;
};

/// Smart point of generic Eigen data object
template <typename T>
using EigenptrT = std::shared_ptr<iEigen<T>>;

/// Implementation of iEigen that assigns Tensor operator to Tensor object
template <typename T, typename EigenSource, typename EigenArgs>
struct EigenTensOp final : public iEigen<T>
{
	EigenTensOp (DimensionsT dims, EigenArgs args,
		std::function<EigenSource(EigenArgs&)> make_base) :
		args_(args), tensorbase_(make_base(args_)), data_(dims) {}

	/// Implementation of iEigen<T>
	void assign (void) override
	{
		data_ = tensorbase_;
	}

	/// Implementation of iEigen<T>
	T* get_ptr (void) override
	{
		return data_.data();
	}

	/// Tensor operator arguments
	EigenArgs args_;

	/// Tensor operator
	EigenSource tensorbase_;

	/// Output tensor data object
	TensorT<T> data_;
};

/// Implementation of iEigen that assigns TensorMap to Tensor object
/// using some custom assignment
template <typename T, typename EigenArgs>
struct EigenAssignTens final : public iEigen<T>
{
	EigenAssignTens (T init_value, DimensionsT dims, EigenArgs args,
		std::function<void(TensorT<T>&,const EigenArgs&)> assign) :
		args_(args), assign_(assign), data_(dims)
	{
		data_.setConstant(init_value);
	}

	/// Implementation of iEigen<T>
	void assign (void) override
	{
		assign_(data_, args_);
	}

	/// Implementation of iEigen<T>
	T* get_ptr (void) override
	{
		return data_.data();
	}

	/// Tensor operator arguments
	EigenArgs args_;

	/// Tensor assignment
	std::function<void(TensorT<T>&,const EigenArgs&)> assign_;

	/// Output tensor data object
	TensorT<T> data_;
};

/// Implementation of iEigen that assigns Matrix operator to Matrix object
template <typename T, typename EigenSource, typename EigenArgs>
struct EigenMatOp final : public iEigen<T>
{
	EigenMatOp (DimensionsT dims, EigenArgs args,
		std::function<EigenSource(EigenArgs&)> make_base) :
		args_(args), matrixbase_(make_base(args_)),
		data_(dims.at(1), dims.at(0)) {}

	/// Implementation of iEigen<T>
	void assign (void) override
	{
		data_ = matrixbase_;
	}

	/// Implementation of iEigen<T>
	T* get_ptr (void) override
	{
		return data_.data();
	}

	/// Matrix operator arguments
	EigenArgs args_;

	/// Matrix operator
	EigenSource matrixbase_;

	/// Output matrix data object
	MatrixT<T> data_;
};

/// Return Eigen Tensor wrapper given output shape,
/// and Eigen operator creation and arguments
template <typename T, typename EigenSource, typename EigenArgs>
inline EigenptrT<T> make_eigentensor (DimensionsT dims, EigenArgs args,
	std::function<EigenSource(EigenArgs&)> make_base)
{
	return std::make_shared<EigenTensOp<T,EigenSource,EigenArgs>>(
		dims, args, make_base);
}

/// Return Eigen Matrix wrapper given output shape,
/// and Eigen operator creation and arguments
template <typename T, typename EigenSource, typename EigenArgs>
inline EigenptrT<T> make_eigenmatrix (DimensionsT dims, EigenArgs args,
	std::function<EigenSource(EigenArgs&)> make_base)
{
	return std::make_shared<EigenMatOp<T,EigenSource,EigenArgs>>(
		dims, args, make_base);
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
		logs::fatal("cannot get matmap from nullptr");
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
		logs::fatal("cannot get tensmap from nullptr");
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

#endif // EIGEN_EIGEN_HPP
