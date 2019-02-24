#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"

#ifndef EAD_EIGEN_HPP
#define EAD_EIGEN_HPP

namespace ead
{

// eigen shape
using DimensionsT = std::array<Eigen::Index,8>;

// 4 base eigen types
template <typename T>
using  MatrixT = Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;

template <typename T>
using TensorT = Eigen::Tensor<T,8>;

template <typename T>
using TensMapT = Eigen::TensorMap<TensorT<T>>;

template <typename T>
using MatMapT = Eigen::Map<MatrixT<T>>;

// conversions between 4 base eigen types
template <typename T>
inline MatMapT<T> tens_to_matmap (TensorT<T>& tens)
{
	return MatMapT<T>(tens.data(),
		tens.dimension(1), tens.dimension(0));
}

template <typename T>
inline MatMapT<T> tensmap_to_matmap (TensMapT<T>& tens)
{
	return MatMapT<T>(tens.data(),
		tens.dimension(1), tens.dimension(0));
}

template <typename T>
inline TensMapT<T> mat_to_tensmap (MatrixT<T>& mat)
{
	return TensMapT<T>(mat.data(),
		mat.cols(), mat.rows(), 1,1,1,1,1,1);
}

template <typename T>
inline TensMapT<T> tens_to_tensmap (TensorT<T>& tens)
{
	return TensMapT<T>(tens.data(), tens.dimensions());
}

// generic eigen bridge wrapper
template <typename T>
struct iEigen
{
	virtual ~iEigen (void) = default;

	virtual void assign (void) = 0;

	virtual TensMapT<T>& get_out (void) = 0;
};

template <typename T>
using EigenptrT = std::shared_ptr<iEigen<T>>;

template <typename T, typename EigenSource>
struct EigenTensOp final : public iEigen<T>
{
	EigenTensOp (DimensionsT dims, EigenSource& base) :
		tensorbase_(base), data_(dims),
		out_(tens_to_tensmap(data_)) {}

	void assign (void) override
	{
		data_ = tensorbase_.reshape(data_.dimensions());
	}

	TensMapT<T>& get_out (void) override
	{
		return out_;
	}

	EigenSource tensorbase_;

	TensorT<T> data_;

	TensMapT<T> out_;
};

template <typename T, typename EigenSource>
struct EigenMatOp final : public iEigen<T>
{
	EigenMatOp (DimensionsT dims, EigenSource& base) :
		matrixbase_(base), data_(dims.at(1), dims.at(0)),
		out_(mat_to_tensmap(data_)) {}

	void assign (void) override
	{
		data_ = matrixbase_;
	}

	TensMapT<T>& get_out (void) override
	{
		return out_;
	}

	EigenSource matrixbase_;

	MatrixT<T> data_;

	TensMapT<T> out_;
};

template <typename T, typename EigenSource>
inline EigenptrT<T> make_eigentensor (DimensionsT dims, EigenSource source)
{
	return std::make_shared<EigenTensOp<T,EigenSource>>(dims, source);
}

template <typename T, typename EigenSource>
inline EigenptrT<T> make_eigenmatrix (DimensionsT dims, EigenSource source)
{
	return std::make_shared<EigenMatOp<T,EigenSource>>(dims, source);
}

}

#endif // EAD_EIGEN_HPP
