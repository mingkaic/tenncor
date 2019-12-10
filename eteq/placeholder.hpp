#include "eteq/signature.hpp"
#include "eteq/variable.hpp"

#ifndef ETEQ_PLACEHOLDER_HPP
#define ETEQ_PLACEHOLDER_HPP

namespace eteq
{

/// A "dynamic" function that takes nodes OR
/// shaped array/eigen tensors as a single input
/// If its input are shaped array/eigen tensors,
/// then wrap it in variable and assign as child
/// Otherwise take node as its child
template <typename T>
struct Placeholder final : public iLink<T>
{
	Placeholder (teq::ShapeSignature shape, std::string label = "") :
		label_(label), shape_(shape) {}

	/// Return deep copy of this Functor
	Placeholder<T>* clone (void) const
	{
		return static_cast<Placeholder<T>*>(clone_impl());
	}

	void assign (const LinkptrT<T>& input)
	{
		teq::Shape shape = input->shape();
		if (false == shape.compatible_after(shape_, 0))
		{
			logs::fatalf("assigning data shaped %s to tensor %s",
				shape.to_string().c_str(), shape_.to_string().c_str());
		}
		content_ = input;
	}

	void assign (const teq::TensptrT& input)
	{
		assign(to_link<T>(input));
	}

	void assign (eigen::TensMapT<T>& input)
	{
		teq::Shape shape = eigen::get_shape(input);
		if (auto var = nullptr == content_ ? nullptr :
			dynamic_cast<Variable<T>*>(content_->get_tensor().get()))
		{
			if (false == shape.compatible_after(shape_, 0))
			{
				logs::fatalf("assigning data shaped %s to tensor %s",
					shape.to_string().c_str(), shape_.to_string().c_str());
			}
			var->assign(input);
		}
		else
		{
			assign(make_variable<T>(input.data(), shape, label_));
		}
	}

	void assign (eigen::TensorT<T>& input)
	{
		auto tensmap = eigen::tens_to_tensmap(input);
		assign(tensmap);
	}

	void assign (teq::ShapedArr<T>& sarr)
	{
		if (auto var = nullptr == content_ ? nullptr :
			dynamic_cast<Variable<T>*>(content_->get_tensor().get()))
		{
			if (false == sarr.shape_.compatible_after(shape_, 0))
			{
				logs::fatalf("assigning data shaped %s to tensor %s",
					sarr.shape_.to_string().c_str(), shape_.to_string().c_str());
			}
			var->assign(sarr);
		}
		else
		{
			assign(make_variable<T>(
				sarr.data_.data(), sarr.shape_, label_));
		}
	}

	/// Implementation of iAttributed
	const marsh::iObject* get_attr (std::string attr_name) const override
	{
		return nullptr;
	}

	/// Implementation of iAttributed
	std::vector<std::string> ls_attrs (void) const override
	{
		return {};
	}

	/// Implementation of iEdge
	teq::TensptrT get_tensor (void) const override
	{
		if (nullptr == content_)
		{
			logs::fatal("cannot get tensor of unassigned placeholder");
		}
		return content_->get_tensor();
	}

	/// Implementation of iLink<T>
	T* data (void) const override
	{
		if (nullptr == content_)
		{
			logs::fatal("cannot get data of unassigned placeholder");
		}
		return content_->data();
	}

	/// Implementation of iLink<T>
	bool has_data (void) const override
	{
		if (nullptr == content_)
		{
			return false;
		}
		return content_->has_data();
	}

	/// Implementation of iSignature<T>
	std::string to_string (void) const override
	{
		return label_;
	}

	/// Implementation of iSignature<T>
	teq::ShapeSignature shape_sign (void) const override
	{
		return shape_;
	}

	/// Implementation of iSignature<T>
	bool is_real (void) const override
	{
		return false;
	}

	LinkptrT<T> get_child (void) const
	{
		if (nullptr == content_)
		{
			logs::fatal("cannot get child of unassigned placeholder");
		}
		return content_;
	}

	std::string label_;

private:
	Placeholder (const Placeholder<T>& other) = default;

	iLink<T>* clone_impl (void) const override
	{
		return new Placeholder<T>(*this);
	}

	void subscribe (Functor<T>* parent) override {}

	void unsubscribe (Functor<T>* parent) override {}

	teq::ShapeSignature shape_;

	LinkptrT<T> content_ = nullptr;
};

/// Smart pointer of placeholder nodes to preserve assign functions
template <typename T>
using PlaceptrT = std::shared_ptr<Placeholder<T>>;

}

#endif // ETEQ_PLACEHOLDER_HPP
