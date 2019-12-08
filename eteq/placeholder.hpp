#include "teq/shaped_arr.hpp"
#include "teq/iopfunc.hpp"

#include "eteq/ifunctor.hpp"
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
struct Placeholder final : public iFunctor<T>
{
	static Placeholder<T>* get (
		teq::ShapeSignature shape, std::string label = "")
	{
		return new Placeholder<T>(shape, label);
	}

	/// Return deep copy of this Functor
	Placeholder<T>* clone (void) const
	{
		return static_cast<Placeholder<T>*>(clone_impl());
	}

	/// Return move of this Placeholder
	Placeholder<T>* move (void)
	{
		return new Placeholder<T>(std::move(*this));
	}

	Placeholder<T>& operator = (const Placeholder<T>& other) = delete;

	Placeholder<T>& operator = (Placeholder<T>&& other) = delete;

	void assign (const LinkptrT<T>& input)
	{
		teq::Shape shape = input->shape();
		if (false == shape.compatible_after(shape_, 0))
		{
			logs::fatalf("assigning data shaped %s to tensor %s",
				shape.to_string().c_str(), shape_.to_string().c_str());
		}
		if (nullptr != content_)
		{
			for (auto& parent : this->subs_)
			{
				parent->uninitialize();
			}
		}
		content_ = input;
	}

	void assign (const teq::TensptrT& input)
	{
		teq::Shape shape = input->shape();
		if (false == shape.compatible_after(shape_, 0))
		{
			logs::fatalf("assigning data shaped %s to tensor %s",
				shape.to_string().c_str(), shape_.to_string().c_str());
		}
		if (nullptr != content_)
		{
			for (auto& parent : this->subs_)
			{
				parent->uninitialize();
			}
		}
		content_ = to_node<T>(input);
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
		assign(eigen::tens_to_tensmap(input));
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

	/// Implementation of iTensor
	void accept (teq::iTraveler& visiter) override
	{
		visiter.visit(this);
	}

	/// Implementation of iTensor
	teq::Shape shape (void) const override
	{
		// if (nullptr == content_)
		// {
		// 	logs::fatal("cannot get shape of unassigned placeholder");
		// }
		// return content_->shape();
		return shape_;
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return label_;
	}

	/// Implementation of iData
	void* data (void) override
	{
		if (nullptr == content_)
		{
			logs::fatal("cannot get data of unassigned placeholder");
		}
		return content_->data();
	}

	/// Implementation of iData
	const void* data (void) const override
	{
		if (nullptr == content_)
		{
			logs::fatal("cannot get data of unassigned placeholder");
		}
		return content_->data();
	}

	/// Implementation of iData
	size_t nbytes (void) const override
	{
		if (nullptr == content_)
		{
			logs::fatal("cannot get nbytes of unassigned placeholder");
		}
		return sizeof(T) * shape().n_elems();
	}

	/// Implementation of iFunctor
	teq::Opcode get_opcode (void) const override
	{
		return teq::Opcode{"PLACEHOLDER", egen::PLACEHOLDER};
	}

	/// Implementation of iFunctor
	teq::EdgeRefsT get_children (void) const override
	{
		if (nullptr == content_)
		{
			logs::fatal("cannot get children of unassigned placeholder");
		}
		return {*content_};
	}

	/// Implementation of iFunctor
	marsh::iObject* get_attr (std::string attr_name) const override
	{
		return nullptr;
	}

	/// Implementation of iFunctor
	std::vector<std::string> ls_attrs (void) const override
	{
		return {};
	}

	/// Implementation of iFunctor
	void update_child (teq::TensptrT arg, size_t index) override
	{
		if (index > 0)
		{
			logs::fatalf("cannot modify argument %d in placeholder", index);
		}
		assign(to_node<T>(arg));
	}

	/// Implementation of iOperableFunc
	void update (void) override
	{
		if (nullptr == content_)
		{
			logs::fatal("cannot update unassigned placeholder");
		}
		content_->update();
	}

	bool is_uninit (void) const override
	{
		return nullptr == content_;
	}

	std::string label_;

private:
	Placeholder (teq::ShapeSignature shape, std::string label) :
		label_(label),
		shape_(std::vector<teq::DimT>(shape.begin(), shape.end())) {}

	Placeholder (const Placeholder<T>& other) = default;

	Placeholder (Placeholder<T>&& other) = default;

	teq::iTensor* clone_impl (void) const override
	{
		return new Placeholder<T>(*this);
	}

	// teq::ShapeSignature shape_;
	teq::Shape shape_;
	// todo: replace shape with signature when functors also support incomplete shapes

	std::shared_ptr<iLink<T>> content_ = nullptr;
};

/// Smart pointer of placeholder nodes to preserve assign functions
template <typename T>
using PlaceptrT = std::shared_ptr<Placeholder<T>>;

template <typename T>
PlaceptrT<T> make_placeholder (teq::ShapeSignature shape,
	std::string label = "")
{
	return PlaceptrT<T>(Placeholder<T>::get(shape, label));
}

}

#endif // ETEQ_PLACEHOLDER_HPP
