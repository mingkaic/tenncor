///
/// variable.hpp
/// eteq
///
/// Purpose:
/// Define data structures for owning, and passing
///	generalized and type-specific data
///

#include "eteq/ileaf.hpp"
#include "eteq/inode.hpp"
#include "eteq/shaped_arr.hpp"

#ifndef ETEQ_VARIABLE_HPP
#define ETEQ_VARIABLE_HPP

namespace eteq
{

/// Leaf node implementation containing mutable Eigen data
template <typename T>
struct Variable final : public iLeaf<T>
{
	/// Return Variable given raw pointer array whose size is denoted by shape
	static Variable<T>* get (T* ptr, teq::Shape shape, std::string label = "");

	/// Return zero-initialized Variable of specified shape
	static Variable<T>* get (teq::Shape shape, std::string label = "")
	{
		return Variable<T>::get(std::vector<T>(shape.n_elems(), 0),
			shape, label);
	}

	/// Return scalar-initialized Variable of specified shape
	static Variable<T>* get (T scalar, teq::Shape shape, std::string label = "")
	{
		if (label.empty())
		{
			label = fmts::to_string(scalar);
		}
		return Variable<T>::get(std::vector<T>(shape.n_elems(),scalar),
			shape, label);
	}

	/// Return Variable whose data is initialized by vector data
	static Variable<T>* get (std::vector<T> data, teq::Shape shape,
		std::string label = "")
	{
		if (data.size() != shape.n_elems())
		{
			logs::fatalf("cannot create variable with data size %d "
				"against shape %s", data.size(), shape.to_string().c_str());
		}
		return Variable<T>::get(data.data(), shape, label);
	}

	/// Return deep copy of other Variable
	static Variable<T>* get (const Variable<T>& other)
	{
		return new Variable<T>(other);
	}

	/// Return move of other Variable
	static Variable<T>* get (Variable<T>&& other)
	{
		return new Variable<T>(std::move(other));
	}

	Variable<T>& operator = (const Variable<T>& other) = default;

	Variable<T>& operator = (Variable<T>&& other) = default;

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
	Variable<T>& operator = (const TensorT<T>& input)
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
		this->data_ = make_tensmap<T>(data.data(), shape);
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return label_;
	}

	/// Implementation of iLeaf
	bool is_const (void) const override
	{
		return false;
	}

	/// Label for distinguishing variable nodes
	std::string label_; // todo: make private

private:
	Variable (T* data, teq::Shape shape, std::string label) :
		iLeaf<T>(data, shape), label_(label) {}

	Variable (const Variable<T>& other) = default;

	Variable (Variable<T>&& other) = default;

};

/// Variable's node wrapper
template <typename T>
struct VariableNode final : public iNode<T>
{
	VariableNode (std::shared_ptr<Variable<T>> var) : var_(var) {}

	/// Return deep copy of this instance (with a copied variable)
	VariableNode<T>* clone (void) const
	{
		return static_cast<VariableNode<T>*>(clone_impl());
	}

	/// Implementation of iNode<T>
	T* data (void) override
	{
		return (T*) var_->data();
	}

	/// Implementation of iNode<T>
	void update (void) override {}

	/// Implementation of iNode<T>
	teq::TensptrT get_tensor (void) const override
	{
		return var_;
	}

	/// Wrapper around variable assign of the same signature
	void assign (const T* input, teq::Shape shape)
	{
		var_->assign(input, egen::get_type<T>(), shape);
	}

	/// Assign Eigen tensor map to variable's internal data
	void assign (const TensMapT<T>* tensmap)
	{
		var_->assign(tensmap->data(), egen::get_type<T>(), get_shape(*tensmap));
	}

	/// Assign ShapedArr representation to variable's internal data
	void assign (const ShapedArr<T>& arr)
	{
		var_->assign(arr.data_.data(), egen::get_type<T>(), arr.shape_);
	}

protected:
	iNode<T>* clone_impl (void) const override
	{
		return new VariableNode(
			std::shared_ptr<Variable<T>>(Variable<T>::get(*var_)));
	}

private:
	std::shared_ptr<Variable<T>> var_;
};

template <typename T>
Variable<T>* Variable<T>::get (T* ptr, teq::Shape shape, std::string label)
{
	static bool registered = register_builder<Variable<T>,T>(
		[](teq::TensptrT tens)
		{
			return std::make_shared<VariableNode<T>>(
				std::static_pointer_cast<Variable<T>>(tens));
		});
	assert(registered);

	return new Variable<T>(ptr, shape, label);
}

/// Smart pointer of variable nodes to preserve assign functions
template <typename T>
using VarptrT = std::shared_ptr<VariableNode<T>>;

/// Return Node smart pointer of Variable smart pointer
template <typename T>
NodeptrT<T> convert_to_node (VarptrT<T> var)
{
	return std::static_pointer_cast<iNode<T>>(var);
}

/// Return variable node given scalar and shape
template <typename T>
VarptrT<T> make_variable_scalar (T scalar, teq::Shape shape, std::string label = "")
{
	return std::make_shared<VariableNode<T>>(
		std::shared_ptr<Variable<T>>(Variable<T>::get(scalar, shape, label))
	);
}

/// Return zero-initialized variable node of specified shape
template <typename T>
VarptrT<T> make_variable (teq::Shape shape, std::string label = "")
{
	return std::make_shared<VariableNode<T>>(
		std::shared_ptr<Variable<T>>(Variable<T>::get(shape, label))
	);
}

/// Return variable node given raw array and shape
template <typename T>
VarptrT<T> make_variable (T* data, teq::Shape shape, std::string label = "")
{
	return std::make_shared<VariableNode<T>>(
		std::shared_ptr<Variable<T>>(Variable<T>::get(data, shape, label))
	);
}

}

#endif // ETEQ_VARIABLE_HPP
