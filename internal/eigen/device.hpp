
#ifndef EIGEN_DEVICE_HPP
#define EIGEN_DEVICE_HPP

#include "internal/eigen/convert.hpp"
#include "internal/eigen/observable.hpp"
#include "internal/eigen/memory.hpp"

namespace eigen
{

struct iEigen : public teq::iDeviceRef
{
	virtual ~iEigen (void) = default;

	virtual void assign (size_t ttl, iRuntimeMemory& runtime) = 0;

	virtual bool valid_for (size_t desired_ttl) const = 0;

	virtual void extend_life (size_t ttl) = 0;
};

/// Smart point of generic Eigen data object
using EigenptrT = std::shared_ptr<iEigen>;

struct iPermEigen : public iEigen
{
	virtual ~iPermEigen (void) = default;

	bool valid_for (size_t) const override
	{
		return true;
	}

	void extend_life (size_t) override {}
};

#ifdef PERM_OP

template <typename T, typename INTYPE=T>
struct PermTensOp final : public iPermEigen
{
	using OpF = std::function<void(TensorT<T>&,const std::vector<TensMapT<INTYPE>>&)>;

	PermTensOp (const teq::Shape& outshape, const teq::CTensT& args,
		OpF op, T init = 0) :
		data_(shape_convert(outshape)), op_(op), args_(args)
	{
		data_.setConstant(init);
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

	teq::Once<void*> odata (void) override
	{
		teq::Once<void*> out(data());
		return out;
	}

	teq::Once<const void*> odata (void) const override
	{
		teq::Once<const void*> out(data());
		return out;
	}

	/// Implementation of iEigen
	void assign (size_t, iRuntimeMemory&) override
	{
		std::vector<TensMapT<INTYPE>> args;
		std::vector<teq::Once<const void*>> onces;
		args.reserve(args_.size());
		onces.reserve(args_.size());
		for (auto& arg : args_)
		{
			teq::Once<const void*> argdata = arg->device().odata();
			assert(nullptr != argdata.get());
			args.push_back(make_tensmap((INTYPE*) argdata.get(), arg->shape()));
			onces.push_back(std::move(argdata));
		}
		op_(data_, args);
	}

private:
	TensorT<T> data_;

	OpF op_;

	/// Tensor operator arguments
	teq::CTensT args_;
};

template <typename T, typename INTYPE=T>
struct PermMatOp final : public iPermEigen
{
	using OpF = std::function<void(MatrixT<T>&,const std::vector<MatMapT<INTYPE>>&)>;

	PermMatOp (const teq::Shape& outshape, const teq::CTensT& args,
		OpF op, T init = 0) :
		data_(outshape.at(1), outshape.at(0)), op_(op), args_(args)
	{
		data_.setConstant(init);
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

	teq::Once<void*> odata (void) override
	{
		teq::Once<void*> out(data());
		return out;
	}

	teq::Once<const void*> odata (void) const override
	{
		teq::Once<const void*> out(data());
		return out;
	}

	/// Implementation of iEigen
	void assign (size_t, iRuntimeMemory&) override
	{
		std::vector<MatMapT<INTYPE>> args;
		std::vector<teq::Once<const void*>> onces;
		args.reserve(args_.size());
		onces.reserve(args_.size());
		for (auto& arg : args_)
		{
			teq::Once<const void*> argdata = arg->device().odata();
			assert(nullptr != argdata.get());
			args.push_back(make_matmap((INTYPE*) argdata.get(), arg->shape()));
			onces.push_back(std::move(argdata));
		}
		op_(data_, args);
	}

private:
	MatrixT<T> data_;

	OpF op_;

	/// Tensor operator arguments
	teq::CTensT args_;
};

#endif

struct CacheEigen final : public iPermEigen
{
	CacheEigen (EigenptrT data) : data_(data)
	{
		if (nullptr == data)
		{
			global::fatal("cannot cache null eigen device ref");
		}
	}

	/// Implementation of iDeviceRef
	void* data (void) override
	{
		return data_->data();
	}

	/// Implementation of iDeviceRef
	const void* data (void) const override
	{
		return data_->data();
	}

	teq::Once<void*> odata (void) override
	{
		teq::Once<void*> out(data());
		return out;
	}

	teq::Once<const void*> odata (void) const override
	{
		teq::Once<const void*> out(data());
		return out;
	}

	/// Implementation of iEigen
	void assign (size_t ttl, iRuntimeMemory& runtime) override
	{
		data_->assign(ttl, runtime);
	}

private:
	EigenptrT data_;
};

/// Source device reference useful for leaves
template <typename T>
struct SrcRef final : public iPermEigen
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

	teq::Once<void*> odata (void) override
	{
		teq::Once<void*> out(data());
		return out;
	}

	teq::Once<const void*> odata (void) const override
	{
		teq::Once<const void*> out(data());
		return out;
	}

	/// Implementation of iEigen
	void assign (size_t, iRuntimeMemory&) override {}

	void assign (const TensMapT<T>& input)
	{
		data_ = input;
	}

	void assign (const TensorT<T>& input)
	{
		data_ = input;
	}

private:
	/// Data Source
	TensorT<T> data_;
};

template <typename T>
struct iTmpEigen : public iEigen
{
	virtual ~iTmpEigen (void)
	{
		data_.expire();
	}

	/// Implementation of iDeviceRef
	void* data (void) override
	{
		return data_.get();
	}

	/// Implementation of iDeviceRef
	const void* data (void) const override
	{
		return data_.get();
	}

	teq::Once<void*> odata (void) override
	{
		teq::Once<void*> out(data(), [this]{ data_.tick(); });
		return out;
	}

	teq::Once<const void*> odata (void) const override
	{
		teq::Once<const void*> out(data(), [this]{ data_.tick(); });
		return out;
	}

	bool valid_for (size_t desired_ttl) const override
	{
		return desired_ttl <= data_.get_ttl();
	}

	void extend_life (size_t ttl) override
	{
		data_.extend_life(ttl);
	}

protected:
	mutable Expirable<T> data_;
};

template <typename T, typename INTYPE=T>
struct TensOp final : public iTmpEigen<T>
{
	using OpF = std::function<void(TensMapT<T>&,const std::vector<TensMapT<INTYPE>>&)>;

	TensOp (const teq::Shape& outshape, const teq::CTensT& args, OpF op) :
		outshape_(outshape), op_(op), args_(args) {}

	/// Implementation of iEigen
	void assign (size_t ttl, iRuntimeMemory& runtime) override
	{
		if (this->data_.is_expired())
		{
			this->data_.borrow(runtime, outshape_.n_elems(), ttl);
		}
		else
		{
			this->data_.extend_life(ttl);
		}
		auto out = make_tensmap(this->data_.get(), outshape_);
		std::vector<TensMapT<INTYPE>> args;
		std::vector<teq::Once<const void*>> onces;
		args.reserve(args_.size());
		onces.reserve(args_.size());
		for (auto& arg : args_)
		{
			teq::Once<const void*> argdata = arg->device().odata();
			assert(nullptr != argdata.get());
			args.push_back(make_tensmap((INTYPE*) argdata.get(), arg->shape()));
			onces.push_back(std::move(argdata));
		}
		op_(out, args);
	}

private:
	teq::Shape outshape_;

	OpF op_;

	/// Tensor operator arguments
	teq::CTensT args_;
};

template <typename T, typename INTYPE=T>
struct MatOp final : public iTmpEigen<T>
{
	using OpF = std::function<void(MatMapT<T>&,const std::vector<MatMapT<INTYPE>>&)>;

	MatOp (const teq::Shape& outshape, const teq::CTensT& args, OpF op) :
		outshape_(outshape), op_(op), args_(args) {}

	/// Implementation of iEigen
	void assign (size_t ttl, iRuntimeMemory& runtime) override
	{
		if (this->data_.is_expired())
		{
			this->data_.borrow(runtime, outshape_.n_elems(), ttl);
		}
		else
		{
			this->data_.extend_life(ttl);
		}
		auto out = make_matmap(this->data_.get(), outshape_);
		std::vector<MatMapT<INTYPE>> args;
		std::vector<teq::Once<const void*>> onces;
		args.reserve(args_.size());
		onces.reserve(args_.size());
		for (auto& arg : args_)
		{
			teq::Once<const void*> argdata = arg->device().odata();
			assert(nullptr != argdata.get());
			args.push_back(make_matmap((INTYPE*) argdata.get(), arg->shape()));
			onces.push_back(std::move(argdata));
		}
		op_(out, args);
	}

private:
	teq::Shape outshape_;

	OpF op_;

	/// Tensor operator arguments
	teq::CTensT args_;
};

struct iRefEigen : public iEigen
{
	iRefEigen (teq::iTensor& ref) : ref_(&ref) {}

	virtual ~iRefEigen (void) = default;

	teq::Once<void*> odata (void) override
	{
		teq::Once<void*> out(data(),
		[this]
		{
			if (0 == (--this->ref_ttl_))
			{
				this->ref_->device().odata();
			}
		});
		return out;
	}

	teq::Once<const void*> odata (void) const override
	{
		teq::Once<const void*> out(data(),
		[this]
		{
			if (0 == (--this->ref_ttl_))
			{
				this->ref_->device().odata();
			}
		});
		return out;
	}

	bool valid_for (size_t desired_ttl) const override
	{
		return desired_ttl <= ref_ttl_;
	}

	void extend_life (size_t ttl) override
	{
		if (ref_ttl_ < ttl)
		{
			ref_ttl_ = ttl;
		}
	}

protected:
	teq::iTensor* ref_;

private:
	// Ref has an independent from its reference, to accurately reflect dependencies
	mutable size_t ref_ttl_ = 0;
};

/// Directly proxy the iDeviceRef of an argument tensor and perform pointer manipulation if available
struct TensRef final : public iRefEigen
{
	TensRef (teq::iTensor& ref) : iRefEigen(ref) {}

	/// Implementation of iDeviceRef
	void* data (void) override
	{
		return this->ref_->device().data();
	}

	/// Implementation of iDeviceRef
	const void* data (void) const override
	{
		return this->ref_->device().data();
	}

	/// Implementation of iEigen
	void assign (size_t ttl, iRuntimeMemory&) override
	{
		extend_life(ttl);
	}
};

// Same as TensRef except increment output pointers by specific increment of template type
template <typename T>
struct UnsafeTensRef final : public iRefEigen
{
	UnsafeTensRef (teq::iTensor& ref, size_t ptr_incrs) :
		iRefEigen(ref), incrs_(ptr_incrs)
	{
		if (incrs_ == 0)
		{
			// this is a message intended for devs (todo: move to compile time)
			global::warn("Creating UnsafeTensRef without pointer manipulation. "
				"Please use TensRef instead");
		}
	}

	/// Implementation of iDeviceRef
	void* data (void) override
	{
		return manipulate(this->ref_->device().data());
	}

	/// Implementation of iDeviceRef
	const void* data (void) const override
	{
		return manipulate(this->ref_->device().data());
	}

	/// Implementation of iEigen
	void assign (size_t ttl, iRuntimeMemory&) override
	{
		extend_life(ttl);
	}

private:
	void* manipulate (void* in) const
	{
		return ((T*) in) + incrs_;
	}

	const void* manipulate (const void* in) const
	{
		return ((const T*) in) + incrs_;
	}

	size_t incrs_;
};

template <typename T>
struct TensAssign final : public iRefEigen
{
	using AssignF = std::function<void(TensMapT<T>&,const TensMapT<T>&)>;

	TensAssign (teq::iTensor& target, const teq::iTensor& arg, AssignF assign) :
		iRefEigen(target), arg_(&arg), assign_(assign) {}

	/// Implementation of iDeviceRef
	void* data (void) override
	{
		return this->ref_->device().data();
	}

	/// Implementation of iDeviceRef
	const void* data (void) const override
	{
		return this->ref_->device().data();
	}

	/// Implementation of iEigen
	void assign (size_t ttl, iRuntimeMemory&) override
	{
		extend_life(ttl);
		auto next_version = arg_->get_meta().state_version() + 1;
		static_cast<iMutableLeaf*>(this->ref_)->upversion(next_version);
		auto dst_data = (T*) this->ref_->device().data();
		auto src_data = (T*) arg_->device().data();
		auto out = make_tensmap(dst_data, this->ref_->shape());
		assign_(out, make_tensmap(src_data, arg_->shape()));
	}

private:
	/// Assignment argument
	const teq::iTensor* arg_;

	AssignF assign_;
};

static RuntimeMemory global_runtime; // temporary workaround

struct Device final : public teq::iDevice
{
	Device (size_t max_version = std::numeric_limits<size_t>::max()) :
		max_version_(max_version) {}

	void calc (teq::iTensor& tens, size_t cache_ttl) override
	{
		auto& obs = static_cast<Observable&>(tens);
		auto& obsdev = static_cast<iEigen&>(tens.device());
		size_t valid_ttl = obs.nsubs() + cache_ttl;
		// always assign when device is stateless (even if version is the same)
		if (obs.prop_version(max_version_) ||
			nullptr == obsdev.data())
		{
			obsdev.assign(std::max<size_t>(1, valid_ttl), global_runtime);
		}
		else if (false == obsdev.valid_for(valid_ttl))
		{
			obsdev.extend_life(valid_ttl);
		}
	}

	size_t max_version_;
};

}

#endif // EIGEN_DEVICE_HPP
