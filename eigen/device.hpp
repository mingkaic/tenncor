#ifndef EIGEN_DEVICE_HPP
#define EIGEN_DEVICE_HPP

#include "teq/isession.hpp"
#include "teq/config.hpp"

#include "eigen/eigen.hpp"

namespace eigen
{

const std::string device_key = "device";

struct iEigen : public teq::iDeviceRef
{
	virtual void assign (void) = 0;
};

/// Smart point of generic Eigen data object
using EigenptrT = std::shared_ptr<iEigen>;

template <typename T>
struct SrcRef final : public iEigen
{
	SrcRef (T* data, teq::Shape shape) :
		data_(eigen::make_tensmap(data, shape)) {}

	/// Implementation of iDeviceRef
	void* data (void) override
	{
		return data_.data();
	}

	/// Implementation of iDeviceRef
	const void* data (void) const override
	{
		return data_.data();
	}

	/// Implementation of iEigen
	void assign (void) override {}

	/// Data Source
	eigen::TensorT<T> data_;
};

template <typename T>
struct PtrRef final : public iEigen
{
	PtrRef (T* ref) : ref_(ref) {}

	/// Implementation of iDeviceRef
	void* data (void) override
	{
		return ref_;
	}

	/// Implementation of iDeviceRef
	const void* data (void) const override
	{
		return ref_;
	}

	/// Implementation of iEigen
	void assign (void) override {}

	T* ref_;
};

/// Implementation of iEigen that assigns TensorMap to Tensor object
/// using some custom assignment
template <typename T, typename EigenArgs>
struct TensAssign final : public iEigen
{
	TensAssign (T init_value, DimensionsT dims, EigenArgs args,
		std::function<void(TensorT<T>&,const EigenArgs&)> assign) :
		args_(args), assign_(assign), data_(dims), init_(init_value) {}

	/// Implementation of iDeviceRef
	void* data (void) override
	{
		return data_.data();
	}

	/// Implementation of iDeviceRef
	const void* data (void) const override
	{
		return data_.data();
	}

	/// Implementation of iEigen
	void assign (void) override
	{
		data_.setConstant(init_);
		assign_(data_, args_);
	}

	/// Tensor operator arguments
	EigenArgs args_;

	/// Tensor assignment
	std::function<void(TensorT<T>&,const EigenArgs&)> assign_;

	/// Output tensor data object
	TensorT<T> data_;

	T init_;
};

/// Implementation of iEigen that assigns Tensor operator to Tensor object
template <typename T, typename EigenSource, typename EigenArgs>
struct TensOp final : public iEigen
{
	TensOp (DimensionsT dims, EigenArgs args,
		std::function<EigenSource(EigenArgs&)> make_base) :
		args_(args), tensorbase_(make_base(args_)), data_(dims) {}

	/// Implementation of iDeviceRef
	void* data (void) override
	{
		return data_.data();
	}

	/// Implementation of iDeviceRef
	const void* data (void) const override
	{
		return data_.data();
	}

	/// Implementation of iEigen
	void assign (void) override
	{
		data_ = tensorbase_;
	}

	/// Tensor operator arguments
	EigenArgs args_;

	/// Tensor operator
	EigenSource tensorbase_;

	/// Output tensor data object
	TensorT<T> data_;
};

/// Implementation of iEigen that assigns Matrix operator to Matrix object
template <typename T, typename EigenSource, typename EigenArgs>
struct MatOp final : public iEigen
{
	MatOp (DimensionsT dims, EigenArgs args,
		std::function<EigenSource(EigenArgs&)> make_base) :
		args_(args), matrixbase_(make_base(args_)),
		data_(dims.at(1), dims.at(0)) {}

	/// Implementation of iDeviceRef
	void* data (void) override
	{
		return data_.data();
	}

	/// Implementation of iDeviceRef
	const void* data (void) const override
	{
		return data_.data();
	}

	/// Implementation of iEigen
	void assign (void) override
	{
		data_ = matrixbase_;
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
inline EigenptrT make_eigentensor (DimensionsT dims, EigenArgs args,
	std::function<EigenSource(EigenArgs&)> make_base)
{
	return std::make_shared<TensOp<T,EigenSource,EigenArgs>>(
		dims, args, make_base);
}

/// Return Eigen Matrix wrapper given output shape,
/// and Eigen operator creation and arguments
template <typename T, typename EigenSource, typename EigenArgs>
inline EigenptrT make_eigenmatrix (DimensionsT dims, EigenArgs args,
	std::function<EigenSource(EigenArgs&)> make_base)
{
	return std::make_shared<MatOp<T,EigenSource,EigenArgs>>(
		dims, args, make_base);
}

struct Device final : public teq::iDevice
{
	void calc (teq::iDeviceRef& ref) override
	{
		static_cast<iEigen&>(ref).assign();
	}
};

inline Device& default_device (void)
{
	static Device device;
	return device;
}

inline teq::Session get_session (void)
{
	auto device = static_cast<teq::iDevice*>(
		config::global_config.get_obj(device_key));
	if (nullptr == device)
	{
		device = &default_device();
	}
	return teq::Session(*device);
}

#define DEVICE_INIT(DEVICE_TYPE)::config::global_config.add_entry<DEVICE_TYPE>(eigen::device_key)

}

#endif // EIGEN_DEVICE_HPP
