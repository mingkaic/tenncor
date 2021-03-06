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

/// Leaf node implementation containing mutable Eigen data
template <typename T>
struct Variable final : public eigen::iMutableLeaf
{
	/// Return Variable given raw pointer array whose size is denoted by shape
	static Variable<T>* get (T* ptr, teq::Shape shape,
		std::string label = "", teq::Usage usage = teq::VARUSAGE)
	{
		return new Variable<T>(ptr, shape, label, usage);
	}

	/// Return deep copy of this Variable
	Variable<T>* clone (void) const
	{
		return static_cast<Variable<T>*>(clone_impl());
	}

	/// Return move of this Variable
	Variable<T>* move (void)
	{
		return new Variable<T>(std::move(*this));
	}

	Variable<T>& operator = (const Variable<T>& other) = delete;

	Variable<T>& operator = (Variable<T>&& other) = delete;

	void assign (const eigen::TensMapT<T>& input,
		const global::CfgMapptrT& ctx = global::context())
	{
		size_t last_version = get_lastvers(ctx);
		upversion(last_version + 1);
		this->ref_.assign(input);
	}

	void assign (const eigen::TensorT<T>& input,
		const global::CfgMapptrT& ctx = global::context())
	{
		size_t last_version = get_lastvers(ctx);
		upversion(last_version + 1);
		this->ref_.assign(input);
	}

	/// Assign void pointer of specified data type enum and shape
	void assign (const void* input, egen::_GENERATED_DTYPE dtype,
		teq::Shape shape, const global::CfgMapptrT& ctx = global::context())
	{
		if (false == shape.compatible_after(this->shape_, 0))
		{
			global::fatalf("assigning data shaped %s to tensor %s",
				shape.to_string().c_str(), this->shape_.to_string().c_str());
		}
		size_t nelems = shape.n_elems();
		std::vector<T> data(nelems);
		egen::type_convert(&data[0], input, dtype, nelems);
		assign(eigen::make_tensmap<T>(data.data(), shape), ctx);
	}

	void assign (const T* input, teq::Shape shape,
		const global::CfgMapptrT& ctx = global::context())
	{
		assign(input, egen::get_type<T>(), shape, ctx);
	}

	/// Implementation of iTensor
	teq::Shape shape (void) const override
	{
		return shape_;
	}

	/// Implementation of iTensor
	teq::iDeviceRef& device (void) override
	{
		return ref_;
	}

	/// Implementation of iTensor
	const teq::iDeviceRef& device (void) const override
	{
		return ref_;
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
	Variable (T* data, teq::Shape shape, std::string label, teq::Usage usage) :
		ref_(data, shape), shape_(shape), label_(label), usage_(usage) {}

	Variable (const Variable<T>& other) = default;

	Variable (Variable<T>&& other) = default;

	teq::iTensor* clone_impl (void) const override
	{
		return new Variable<T>(*this);
	}

	/// Data Source
	eigen::SrcRef<T> ref_;

	/// Shape utility to avoid excessive conversion between data_.dimensions()
	teq::Shape shape_;

	/// Variable metadata
	eigen::EMetadata<T> meta_ = eigen::EMetadata<T>(1);

	/// Label for distinguishing variable nodes
	std::string label_;

	teq::Usage usage_;
};

/// Smart pointer of variable nodes to preserve assign functions
template <typename T>
using VarptrT = std::shared_ptr<Variable<T>>;

template <typename T>
using VarptrsT = std::vector<VarptrT<T>>;

}

#endif // ETEQ_VARIABLE_HPP
