#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"

#include "ade/shape.hpp"

#include "ead/generated/data.hpp"

#ifndef EAD_TENSOR_HPP
#define EAD_TENSOR_HPP

namespace ead
{

template <typename T>
using  MatrixT = Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;

template <typename T>
using TensorT = Eigen::Tensor<T,ade::rank_cap>;

template <typename T>
using ScalarT = typename TensorT<T>::Scalar;

template <typename T>
using TensMapT = Eigen::TensorMap<TensorT<T>>;

template <typename T>
using MatMapT = Eigen::Map<MatrixT<T>>;

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

template <typename T>
inline TensorT<T> get_tensor (const ade::Shape& shape)
{
	std::array<Eigen::Index,ade::rank_cap> slist;
	std::copy(shape.begin(), shape.end(), slist.begin());
	TensorT<T> out(slist);
	out.setZero();
	return out;
}

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
	EigenTensOp (ade::Shape shape, EigenSource& base) :
		tensorbase_(base), data_(get_tensor<T>(shape)),
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
inline EigenptrT<T> make_tensop (ade::Shape shape, EigenSource source)
{
	return std::make_shared<EigenTensOp<T,EigenSource>>(shape, source);
}

template <typename T, typename EigenSource>
struct EigenMatOp final : public iEigen<T>
{
	EigenMatOp (ade::Shape shape, EigenSource& base) :
		matrixbase_(base), data_(shape.at(1), shape.at(0)),
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
inline EigenptrT<T> make_matop (ade::Shape shape, EigenSource source)
{
	return std::make_shared<EigenMatOp<T,EigenSource>>(shape, source);
}

template <typename T>
using TensptrT = std::shared_ptr<TensorT<T>>;

template <typename T>
inline TensMapT<T> get_tensmap (T* data, const ade::Shape& shape)
{
	std::array<Eigen::Index,ade::rank_cap> slist;
	std::copy(shape.begin(), shape.end(), slist.begin());
	if (nullptr == data)
	{
		logs::fatal("cannot get tensmap from nullptr");;
	}
	return Eigen::TensorMap<TensorT<T>>(data, slist);
}

template <typename T>
TensptrT<T> get_tensorptr (T* data, const ade::Shape& shape)
{
	std::array<Eigen::Index,ade::rank_cap> slist;
	std::copy(shape.begin(), shape.end(), slist.begin());
	if (nullptr != data)
	{
		return std::make_shared<TensorT<T>>(
			Eigen::TensorMap<TensorT<T>>(data, slist));
	}
	auto out = std::make_shared<TensorT<T>>(slist);
	out->setZero();
	return out;
}

template <typename T>
TensptrT<T> raw_to_tensorptr (void* input,
	age::_GENERATED_DTYPE intype, const ade::Shape& shape)
{
	std::vector<T> data;
	age::type_convert(data, input, intype, shape.n_elems());
	return get_tensorptr(data.data(), shape);
}

template <typename T>
ade::Shape get_shape (const TensorT<T>& tens)
{
	auto slist = tens.dimensions();
	return ade::Shape(std::vector<ade::DimT>(slist.begin(), slist.end()));
}

template <typename T>
ade::Shape get_shape (const TensMapT<T>& tens)
{
	auto slist = tens.dimensions();
	return ade::Shape(std::vector<ade::DimT>(slist.begin(), slist.end()));
}

}

#endif // EAD_TENSOR_HPP
