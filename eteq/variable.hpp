///
/// variable.hpp
/// eteq
///
/// Purpose:
/// Define data structures for owning, and passing
///	generalized and type-specific data
///

#include "teq/shaped_arr.hpp"

#include "eteq/ileaf.hpp"

#ifndef ETEQ_VARIABLE_HPP
#define ETEQ_VARIABLE_HPP

namespace eteq
{

/// Leaf node implementation containing mutable Eigen data
template <typename T>
struct Variable final : public iLeaf<T>
{
	/// Return Variable given raw pointer array whose size is denoted by shape
	static Variable<T>* get (T* ptr, teq::Shape shape,
		std::string label = "", teq::Usage usage = teq::Variable)
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

	/// Assign vectorized data to data source
	Variable<T>& operator = (std::vector<T> input)
	{
		size_t ninput = input.size();
		if (this->shape_.n_elems() != ninput)
		{
			logs::fatalf("cannot assign vector of %d elements to "
				"internal data of shape %s", ninput,
				this->shape_.to_string().c_str());
		}
		T* indata = input.data();
		std::copy(indata, indata + ninput, this->data_.data());
		return *this;
	}

	/// Assign Eigen tensor to internal data object
	Variable<T>& operator = (const eigen::TensorT<T>& input)
	{
		this->data_ = input;
		return *this;
	}

	/// Assign void pointer of specified data type enum and shape
	void assign (const void* input, egen::_GENERATED_DTYPE dtype, teq::Shape shape)
	{
		if (false == shape.compatible_after(this->shape_, 0))
		{
			logs::fatalf("assigning data shaped %s to tensor %s",
				shape.to_string().c_str(), this->shape_.to_string().c_str());
		}
		std::vector<T> data;
		egen::type_convert(data, input, dtype, shape.n_elems());
		this->data_ = eigen::make_tensmap<T>(data.data(), shape);
	}

	void assign (eigen::TensMapT<T>& input)
	{
		this->data_ = input;
	}

	void assign (eigen::TensorT<T>& input)
	{
		this->data_ = input;
	}

	void assign (const T* input, teq::Shape shape)
	{
		assign(input, egen::get_type<T>(), shape);
	}

	void assign (const teq::ShapedArr<T>& arr)
	{
		assign(arr.data_.data(), egen::get_type<T>(), arr.shape_);
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

private:
	Variable (T* data, teq::Shape shape, std::string label, teq::Usage usage) :
		iLeaf<T>(data, shape), label_(label), usage_(usage) {}

	Variable (const Variable<T>& other) = default;

	Variable (Variable<T>&& other) = default;

	teq::iTensor* clone_impl (void) const override
	{
		return new Variable<T>(*this);
	}

	/// Label for distinguishing variable nodes
	std::string label_;

	teq::Usage usage_;
};

/// Smart pointer of variable nodes to preserve assign functions
template <typename T>
using VarptrT = std::shared_ptr<Variable<T>>;

/// Return variable node given scalar and shape
template <typename T>
VarptrT<T> make_variable_scalar (T scalar,
	teq::Shape shape, std::string label = "");

/// Return variable node filled with scalar matching link shape
template <typename T>
VarptrT<T> make_variable_like (T scalar,
	LinkptrT<T> link, std::string label = "");

/// Return zero-initialized variable node of specified shape
template <typename T>
VarptrT<T> make_variable (teq::Shape shape, std::string label = "");

/// Return variable node given raw array and shape
template <typename T>
VarptrT<T> make_variable (T* data, teq::Shape shape, std::string label = "");

}

#endif // ETEQ_VARIABLE_HPP
