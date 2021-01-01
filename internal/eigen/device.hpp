
#ifndef EIGEN_DEVICE_HPP
#define EIGEN_DEVICE_HPP

#include "internal/eigen/convert.hpp"
#include "internal/eigen/observable.hpp"

namespace eigen
{

struct iEigen : public teq::iDeviceRef
{
	virtual ~iEigen (void) = default;

	virtual void assign (void) = 0;
};

/// Smart point of generic Eigen data object
using EigenptrT = std::shared_ptr<iEigen>;

template <typename T>
struct SrcRef final : public iEigen
{
	SrcRef (T* data, teq::Shape shape) :
		data_(make_tensmap(data, shape)) {}

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
	TensorT<T> data_;
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

template <typename T, typename ARGS>
struct TensAssign final : public iEigen
{
	TensAssign (teq::iTensor& target, ARGS args,
		std::function<void(TensorT<T>&,ARGS&)> assign) :
		target_(&target), args_(args), assign_(assign) {}

	/// Implementation of iDeviceRef
	void* data (void) override
	{
		return target_->device().data();
	}

	/// Implementation of iDeviceRef
	const void* data (void) const override
	{
		return target_->device().data();
	}

	/// Implementation of iEigen
	void assign (void) override
	{
		assign_(static_cast<SrcRef<T>&>(target_->device()).data_, args_);
	}

	teq::iTensor* target_;

	/// Assignment arguments
	ARGS args_;

	std::function<void(TensorT<T>&,ARGS&)> assign_;
};

/// Implementation of iEigen that assigns TensorMap to Tensor object
/// using some custom assignment
template <typename T, typename ARGS>
struct TensAccum final : public iEigen
{
	TensAccum (T init_value, DimensionsT dims, ARGS args,
		std::function<void(TensorT<T>&,const ARGS&)> assign) :
		args_(args), assign_(assign), data_(dims), init_(init_value)
	{
		data_.setConstant(0);
	}

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
	ARGS args_;

	/// Tensor assignment
	std::function<void(TensorT<T>&,const ARGS&)> assign_;

	/// Output tensor data object
	TensorT<T> data_;

	T init_;
};

/// Implementation of iEigen that assigns Tensor operator to Tensor object
template <typename T, typename SRC, typename ARGS>
struct TensOp final : public iEigen
{
	TensOp (DimensionsT dims, ARGS args,
		std::function<SRC(ARGS&)> make_base) :
		args_(args), tensorbase_(make_base(args_)), data_(dims)
	{
		data_.setConstant(0);
	}

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
	ARGS args_;

	/// Tensor operator
	SRC tensorbase_;

	/// Output tensor data object
	TensorT<T> data_;
};

/// Implementation of iEigen that assigns Matrix operator to Matrix object
template <typename T, typename SRC, typename ARGS>
struct MatOp final : public iEigen
{
	MatOp (DimensionsT dims, ARGS args,
		std::function<SRC(ARGS&)> make_base) :
		args_(args), matrixbase_(make_base(args_)),
		data_(dims.at(1), dims.at(0))
	{
		data_.setConstant(0);
	}

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
	ARGS args_;

	/// Matrix operator
	SRC matrixbase_;

	/// Output matrix data object
	MatrixT<T> data_;
};

/// Return Eigen Tensor wrapper given output shape,
/// and Eigen operator creation and arguments
template <typename T, typename SRC, typename ARGS>
inline EigenptrT make_eigentensor (DimensionsT dims, ARGS args,
	std::function<SRC(ARGS&)> make_base)
{
	return std::make_shared<TensOp<T,SRC,ARGS>>(
		dims, args, make_base);
}

/// Return Eigen Matrix wrapper given output shape,
/// and Eigen operator creation and arguments
template <typename T, typename SRC, typename ARGS>
inline EigenptrT make_eigenmatrix (DimensionsT dims, ARGS args,
	std::function<SRC(ARGS&)> make_base)
{
	return std::make_shared<MatOp<T,SRC,ARGS>>(
		dims, args, make_base);
}

struct Device final : public teq::iDevice
{
	Device (size_t max_version = std::numeric_limits<size_t>::max()) :
		max_version_(max_version) {}

	void calc (teq::iTensor& tens) override
	{
		auto& obs = static_cast<Observable&>(tens);
		if (obs.prop_version(max_version_))
		{
			static_cast<iEigen&>(tens.device()).assign();
		}
	}

	size_t max_version_;
};

}

#endif // EIGEN_DEVICE_HPP
