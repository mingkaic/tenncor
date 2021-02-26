
#ifndef EIGEN_DEVICE_HPP
#define EIGEN_DEVICE_HPP

#include "internal/eigen/convert.hpp"
#include "internal/eigen/observable.hpp"
#include "internal/eigen/memory.hpp"

namespace eigen
{

using OptSparseT = std::optional<SparseInfo>;

struct iEigen : public teq::iDeviceRef
{
	virtual ~iEigen (void) = default;

	virtual OptSparseT sparse_info (void) const = 0;

	virtual void assign (size_t ttl, RTMemptrT& runtime) = 0;

	virtual bool valid_for (size_t desired_ttl) const = 0;

	virtual void extend_life (size_t ttl) = 0;
};

inline bool is_sparse (const teq::iTensor& tens)
{
	auto dev = dynamic_cast<const iEigen*>(&tens.device());
	return dev != nullptr && bool(dev->sparse_info());
}

inline OptSparseT sparse_info (const teq::iTensor& tens)
{
	if (auto dev = dynamic_cast<const iEigen*>(&tens.device()))
	{
		return dev->sparse_info();
	}
	return OptSparseT();
}

template <typename T>
inline teq::Once<MatMapT<T>> make_matmap (const teq::iTensor& tens)
{
	auto& dev = tens.device();
	teq::Once<const void*> data = dev.odata();
	assert(nullptr != data.get());
	return teq::Once<MatMapT<T>>(make_matmap(
		(T*) data.get(), tens.shape()), std::move(data));
}

template <typename T>
inline teq::Once<SMatMapT<T>> make_smatmap (const teq::iTensor& tens)
{
	auto dev = dynamic_cast<const iEigen*>(&tens.device());
	std::optional<SparseInfo> sinfo;
	if (nullptr != dev)
	{
		sinfo = dev->sparse_info();
	}
	if (false == bool(sinfo))
	{
		global::fatal("cannot make sparse matrix from non-sparse device reference");
	}
	teq::Once<const void*> data = dev->odata();
	assert(nullptr != data.get());
	return teq::Once<SMatMapT<T>>(make_smatmap(
		(T*) data.get(), *sinfo, tens.shape()), std::move(data));
}

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

struct iSrcRef : public iPermEigen
{
	virtual ~iSrcRef (void) = default;

	iSrcRef* clone (void) const
	{
		return clone_impl();
	}

	virtual void assign (const void* ptr,
		egen::_GENERATED_DTYPE dtype,
		const OptSparseT& sparse_info,
		const teq::Shape& shape) = 0;

protected:
	virtual iSrcRef* clone_impl (void) const = 0;
};

using SrcRefptrT = std::unique_ptr<iSrcRef>;

#ifdef PERM_OP

template <typename T, typename INTYPE=T>
struct PermTensOp final : public iPermEigen
{
	using OpF = std::function<void(TensorT<T>&,const std::vector<TensMapT<INTYPE>>&)>;

	PermTensOp (const teq::Shape& outshape, const teq::CTensT& args, OpF op) :
		data_(shape_convert(outshape)), op_(op)
	{
		args_.reserve(args.size());
		for (auto& arg : args)
		{
			teq::Once<const void*> argdata = arg->device().odata();
			assert(nullptr != argdata.get());
			args_.push_back(make_tensmap((INTYPE*) argdata.get(), arg->shape()));
		}
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

	OptSparseT sparse_info (void) const override
	{
		return OptSparseT();
	}

	/// Implementation of iEigen
	void assign (size_t, RTMemptrT&) override
	{
		op_(data_, args_);
	}

private:
	/// Tensor operator arguments
	std::vector<TensMapT<INTYPE>> args_;

	/// Output tensor data object
	TensorT<T> data_;

	OpF op_;
};

template <typename T, typename INTYPE=T>
struct PermMatOp final : public iPermEigen
{
	using OpF = std::function<void(MatrixT<T>&,const std::vector<MatMapT<INTYPE>>&)>;

	PermMatOp (const teq::Shape& outshape, const teq::CTensT& args, OpF op) :
		data_(outshape.at(1), outshape.at(0)), op_(op)
	{
		args_.reserve(args.size());
		for (auto& arg : args)
		{
			teq::Once<const void*> argdata = arg->device().odata();
			assert(nullptr != argdata.get());
			args_.push_back(make_matmap((INTYPE*) argdata.get(), arg->shape()));
		}
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

	OptSparseT sparse_info (void) const override
	{
		return OptSparseT();
	}

	/// Implementation of iEigen
	void assign (size_t, RTMemptrT&) override
	{
		op_(data_, args_);
	}

private:
	/// Matrix operator arguments
	std::vector<MatMapT<INTYPE>> args_;

	/// Output matrix data object
	MatrixT<T> data_;

	OpF op_;
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

	OptSparseT sparse_info (void) const override
	{
		return OptSparseT();
	}

	/// Implementation of iEigen
	void assign (size_t ttl, RTMemptrT& runtime) override
	{
		data_->assign(ttl, runtime);
	}

private:
	EigenptrT data_;
};

/// Source device reference useful for leaves
template <typename T>
struct SrcRef final : public iSrcRef
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

	OptSparseT sparse_info (void) const override
	{
		return OptSparseT();
	}

	/// Implementation of iEigen
	void assign (size_t, RTMemptrT&) override {}

	void assign (const TensMapT<T>& input)
	{
		data_ = input;
	}

	void assign (const TensorT<T>& input)
	{
		data_ = input;
	}

	void assign (const void* ptr,
		egen::_GENERATED_DTYPE dtype,
		const OptSparseT& sparse_info,
		const teq::Shape& shape) override
	{
		if (dtype != egen::get_type<T>())
		{
			size_t nelems = shape.n_elems();
			std::vector<T> data(nelems);
			egen::type_convert(&data[0], ptr, dtype, nelems);
			data_ = eigen::make_tensmap<T>(data.data(), shape);
		}
		else
		{
			data_ = eigen::make_tensmap<T>((T*) ptr, shape);
		}
	}

private:
	iSrcRef* clone_impl (void) const
	{
		return new SrcRef<T>(*this);
	}

	/// Data Source
	TensorT<T> data_;
};

template <typename T>
struct SparseSrcRef final : public iSrcRef
{
	SparseSrcRef (T* data, const SparseInfo& sparse_info, teq::Shape shape) :
		data_(make_smatmap(data, sparse_info, shape)) {}

	/// Implementation of iDeviceRef
	void* data (void) override
	{
		return data_.valuePtr();
	}

	/// Implementation of iDeviceRef
	const void* data (void) const override
	{
		return data_.valuePtr();
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

	OptSparseT sparse_info (void) const override
	{
		return SparseInfo::get<T>(data_);
	}

	/// Implementation of iEigen
	void assign (size_t, RTMemptrT&) override {}

	void assign (const SMatMapT<T>& input)
	{
		data_ = input;
	}

	void assign (const SMatrixT<T>& input)
	{
		data_ = input;
	}

	void assign (const void* ptr,
		egen::_GENERATED_DTYPE dtype,
		const OptSparseT& sparse_info,
		const teq::Shape& shape) override
	{
		assert(bool(sparse_info));
		if (dtype != egen::get_type<T>())
		{
			size_t nelems = shape.n_elems();
			std::vector<T> data(nelems);
			egen::type_convert(&data[0], ptr, dtype, nelems);
			data_ = eigen::make_smatmap<T>(data.data(), *sparse_info, shape);
		}
		else
		{
			data_ = eigen::make_smatmap<T>((T*) ptr, *sparse_info, shape);
		}
	}

private:
	iSrcRef* clone_impl (void) const
	{
		return new SparseSrcRef<T>(*this);
	}

	/// Data Source
	mutable SMatrixT<T> data_;
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

	OptSparseT sparse_info (void) const override
	{
		return OptSparseT();
	}

	/// Implementation of iEigen
	void assign (size_t ttl, RTMemptrT& runtime) override
	{
		if (this->data_.is_expired())
		{
			this->data_.borrow(runtime, outshape_.n_elems(), ttl);
		}
		else
		{
			this->data_.extend_life(ttl);
		}
		// argument transformation is the most expensive operation
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

	OptSparseT sparse_info (void) const override
	{
		return OptSparseT();
	}

	/// Implementation of iEigen
	void assign (size_t ttl, RTMemptrT& runtime) override
	{
		if (this->data_.is_expired())
		{
			this->data_.borrow(runtime, outshape_.n_elems(), ttl);
		}
		else
		{
			this->data_.extend_life(ttl);
		}
		// argument transformation is the most expensive operation
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

template <typename T>
struct SparseMatOp final : public iPermEigen
{
	using OpF = std::function<SMatrixT<T>(const teq::CTensT&)>;

	SparseMatOp (const teq::CTensT& args, OpF op) :
		op_(op), args_(args) {}

	/// Implementation of iDeviceRef
	void* data (void) override
	{
		return data_.valuePtr();
	}

	/// Implementation of iDeviceRef
	const void* data (void) const override
	{
		return data_.valuePtr();
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

	OptSparseT sparse_info (void) const override
	{
		return SparseInfo::get<T>(data_);
	}

	/// Implementation of iEigen
	void assign (size_t ttl, RTMemptrT& runtime) override
	{
		data_ = op_(args_);
	}

private:
	OpF op_;

	mutable SMatrixT<T> data_;

	/// Tensor operator arguments
	teq::CTensT args_;
};

template <typename T>
struct GenericMatOp final : public iTmpEigen<T>
{
	using OpF = std::function<void(MatMapT<T>&,const teq::CTensT&)>;

	GenericMatOp (const teq::Shape& outshape, const teq::CTensT& args, OpF op) :
		outshape_(outshape), op_(op), args_(args) {}

	OptSparseT sparse_info (void) const override
	{
		return OptSparseT();
	}

	/// Implementation of iEigen
	void assign (size_t ttl, RTMemptrT& runtime) override
	{
		if (this->data_.is_expired())
		{
			this->data_.borrow(runtime, outshape_.n_elems(), ttl);
		}
		else
		{
			this->data_.extend_life(ttl);
		}
		// argument transformation is the most expensive operation
		auto out = make_matmap(this->data_.get(), outshape_);
		op_(out, args_);
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

	OptSparseT sparse_info (void) const override
	{
		OptSparseT sinfo;
		if (auto dev = dynamic_cast<iEigen*>(&ref_->device()))
		{
			sinfo = dev->sparse_info();
		}
		return sinfo;
	}

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
	void assign (size_t ttl, RTMemptrT&) override
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
	void assign (size_t ttl, RTMemptrT&) override
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
	void assign (size_t ttl, RTMemptrT&) override
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

template <typename T>
struct MatAssign final : public iRefEigen
{
	using AssignF = std::function<void(MatMapT<T>&,const teq::iTensor&)>;

	MatAssign (teq::iTensor& target, const teq::iTensor& arg, AssignF assign) :
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
	void assign (size_t ttl, RTMemptrT&) override
	{
		extend_life(ttl);
		auto next_version = arg_->get_meta().state_version() + 1;
		static_cast<iMutableLeaf*>(this->ref_)->upversion(next_version);
		auto dst_data = (T*) this->ref_->device().data();
		auto out = make_matmap(dst_data, this->ref_->shape());
		assign_(out, *arg_);
	}

private:
	/// Assignment argument
	const teq::iTensor* arg_;

	AssignF assign_;
};

template <typename T>
struct SparseMatAssign final : public iRefEigen
{
	using AssignF = std::function<SMatrixT<T>(const SMatMapT<T>&,const SMatMapT<T>&)>;

	SparseMatAssign (teq::iTensor& target, const teq::iTensor& arg, AssignF assign) :
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
	void assign (size_t ttl, RTMemptrT&) override
	{
		extend_life(ttl);
		auto next_version = arg_->get_meta().state_version() + 1;
		static_cast<iMutableLeaf*>(this->ref_)->upversion(next_version);
		auto dev = dynamic_cast<SparseSrcRef<T>*>(&this->ref_->device());
		assert(dev != nullptr);
		auto out = make_smatmap<T>(*this->ref_);
		auto src = make_smatmap<T>(*arg_);
		dev->assign(assign_(out.get(), src.get()));
	}

private:
	/// Assignment argument
	const teq::iTensor* arg_;

	AssignF assign_;
};

struct Device final : public teq::iDevice
{
	Device (size_t max_version = std::numeric_limits<size_t>::max()) :
		max_version_(max_version), memory_(std::make_shared<RuntimeMemory>()) {}

	Device (RTMemptrT memory,
		size_t max_version = std::numeric_limits<size_t>::max()) :
		max_version_(max_version), memory_(memory) {}

	void calc (teq::iTensor& tens, size_t cache_ttl) override
	{
		auto& obs = static_cast<Observable&>(tens);
		auto& obsdev = static_cast<iEigen&>(tens.device());
		size_t valid_ttl = obs.nsubs() + cache_ttl;
		// always assign when device is stateless (even if version is the same)
		if (obs.prop_version(max_version_) ||
			nullptr == obsdev.data())
		{
			obsdev.assign(std::max<size_t>(1, valid_ttl), memory_);
		}
		else if (false == obsdev.valid_for(valid_ttl))
		{
			obsdev.extend_life(valid_ttl);
		}
	}

	size_t max_version_;

private:
	RTMemptrT memory_;
};

}

#endif // EIGEN_DEVICE_HPP
