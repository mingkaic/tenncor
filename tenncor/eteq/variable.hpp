///
/// variable.hpp
/// eteq
///
/// Purpose:
/// Define data structures for owning, and passing
///	generalized and type-specific data
///

#ifndef ETEQ_VARIABLE_HPP
#define ETEQ_VARIABLE_HPP

#include "internal/eigen/eigen.hpp"

namespace eteq
{

static inline size_t get_lastvers (const global::CfgMapptrT& ctx)
{
	size_t mvers = 0;
	for (auto& r : get_reg(ctx))
	{
		mvers = std::max(mvers, r.second->get_tensor()->get_meta().state_version());
	}
	return mvers;
}

#define TYPED_SRCREF(REALTYPE)\
	out = std::make_unique<eigen::SrcRef<REALTYPE>>((REALTYPE*) data, shape);

static eigen::SrcRefptrT get_srcref (const void* data,
	egen::_GENERATED_DTYPE dtype, const teq::Shape& shape)
{
	eigen::SrcRefptrT out;
	TYPE_LOOKUP(TYPED_SRCREF, dtype);
	return out;
}

#undef TYPED_SRCREF

/// Leaf node implementation containing mutable Eigen data
struct Variable final : public eigen::iMutableLeaf
{
	/// Return Variable given raw pointer array whose size is denoted by shape
	static Variable* get (const void* ptr, egen::_GENERATED_DTYPE dtype, teq::Shape shape,
		std::string label = "", teq::Usage usage = teq::VARUSAGE)
	{
		return new Variable(ptr, dtype, shape, label, usage);
	}

	/// Return deep copy of this Variable
	Variable* clone (void) const
	{
		return static_cast<Variable*>(clone_impl());
	}

	/// Return move of this Variable
	Variable* move (void)
	{
		return new Variable(std::move(*this));
	}

	Variable (const Variable& other) = delete;

	Variable& operator = (const Variable& other) = delete;

	Variable& operator = (Variable&& other) = delete;

	template <typename T>
	void assign (const eigen::TensorT<T>& input,
		const global::CfgMapptrT& ctx = global::context())
	{
		size_t last_version = get_lastvers(ctx);
		upversion(last_version + 1);
		this->ref_->assign(input.data(), egen::get_type<T>(),
			eigen::OptSparseT(), eigen::get_shape(input));
	}

	template <typename T>
	void assign (const eigen::TensMapT<T>& input,
		const global::CfgMapptrT& ctx = global::context())
	{
		size_t last_version = get_lastvers(ctx);
		upversion(last_version + 1);
		this->ref_->assign(input.data(), egen::get_type<T>(),
			eigen::OptSparseT(), eigen::get_shape(input));
	}

	template <typename T>
	void assign (const eigen::MatBaseT<T>& input,
		const global::CfgMapptrT& ctx = global::context())
	{
		size_t last_version = get_lastvers(ctx);
		upversion(last_version + 1);
		this->ref_->assign(input.data(), egen::get_type<T>(),
			eigen::OptSparseT(), teq::Shape({input.cols(), input.rows()}));
	}

	//template <typename T>
	//void assign (const eigen::SparseBaseT<T>& input,
		//const global::CfgMapptrT& ctx = global::context())
	//{
		//size_t last_version = get_lastvers(ctx);
		//upversion(last_version + 1);
		//this->ref_->assign(input.valuePtr(), egen::get_type<T>(),
			//eigen::OptSparseT({input.innerIndexPtr(), input.outerIndexPtr(),
				//input.nonZeros()}), teq::Shape({input.cols(), input.rows()}));
	//}

	template <typename T>
	void assign (const T* input, teq::Shape shape,
		const global::CfgMapptrT& ctx = global::context())
	{
		assign(input, egen::get_type<T>(), eigen::OptSparseT(), shape, ctx);
	}

	template <typename T>
	void assign (const T* input, const eigen::OptSparseT& sparse_info,
		teq::Shape shape, const global::CfgMapptrT& ctx = global::context())
	{
		assign(input, egen::get_type<T>(), sparse_info, shape, ctx);
	}

	void assign (const void* input, egen::_GENERATED_DTYPE dtype,
		teq::Shape shape, const global::CfgMapptrT& ctx = global::context())
	{
		assign(input, dtype, eigen::OptSparseT(), shape, ctx);
	}

	/// Assign void pointer of specified data type enum and shape
	void assign (const void* input,
		egen::_GENERATED_DTYPE dtype, const eigen::OptSparseT& sparse_info,
		teq::Shape shape, const global::CfgMapptrT& ctx = global::context())
	{
		if (false == shape.compatible_after(this->shape_, 0))
		{
			global::fatalf("assigning data shaped %s to tensor %s",
				shape.to_string().c_str(), this->shape_.to_string().c_str());
		}
		size_t last_version = get_lastvers(ctx);
		upversion(last_version + 1);
		this->ref_->assign(input, dtype, sparse_info, shape);
	}

	/// Implementation of iTensor
	teq::Shape shape (void) const override
	{
		return shape_;
	}

	/// Implementation of iTensor
	teq::iDeviceRef& device (void) override
	{
		return *ref_;
	}

	/// Implementation of iTensor
	const teq::iDeviceRef& device (void) const override
	{
		return *ref_;
	}

	/// Implementation of iTensor
	const teq::iMetadata& get_meta (void) const override
	{
		return meta_;
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return label_;
	}

	/// Implementation of iLeaf
	teq::Usage get_usage (void) const override
	{
		return usage_;
	}

	/// Implementation of iMutableLeaf
	void upversion (size_t version) override
	{
		meta_.version_ = std::max(meta_.version_, version);
	}

private:
	Variable (const void* data, egen::_GENERATED_DTYPE dtype,
		teq::Shape shape, std::string label, teq::Usage usage) :
		ref_(get_srcref(data, dtype, shape)), shape_(shape),
		meta_(dtype, 1), label_(label), usage_(usage) {}

	Variable (Variable&& other) = default;

	teq::iTensor* clone_impl (void) const override
	{
		return new Variable(device().data(),
			(egen::_GENERATED_DTYPE) meta_.type_code(),
			shape_, label_, usage_);
	}

	/// Data Source
	eigen::SrcRefptrT ref_;

	/// Shape utility to avoid excessive conversion between data_.dimensions()
	teq::Shape shape_;

	/// Variable metadata
	eigen::EMetadata2 meta_;

	/// Label for distinguishing variable nodes
	std::string label_;

	teq::Usage usage_;
};

/// Smart pointer of variable nodes to preserve assign functions
using VarptrT = std::shared_ptr<Variable>;

using VarptrsT = std::vector<VarptrT>;

}

#endif // ETEQ_VARIABLE_HPP
