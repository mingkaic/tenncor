#include "teq/shaped_arr.hpp"
#include "teq/iopfunc.hpp"

#include "eigen/generated/opcode.hpp"

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
struct Placeholder final : public teq::iOperableFunc, public Observable<Functor<T>*>
{
	static Placeholder<T>* get (
		teq::ShapeSignature shape, std::string label = "");

	/// Return deep copy of other Placeholder
	static Placeholder<T>* get (const Placeholder<T>& other)
	{
		return new Placeholder<T>(other);
	}

	/// Return move of other Placeholder
	static Placeholder<T>* get (Placeholder<T>&& other)
	{
		return new Placeholder<T>(std::move(other));
	}

	Placeholder<T>& operator = (const Placeholder<T>& other) = default;

	Placeholder<T>& operator = (Placeholder<T>&& other) = default;

	void assign (const NodeptrT<T>& input)
	{
		teq::Shape shape = input->shape();
		if (false == shape.compatible_after(shape_, 0))
		{
			logs::fatalf("assigning data shaped %s to tensor %s",
				shape.to_string().c_str(), shape_.to_string().c_str());
		}
		if (nullptr == content_)
		{
			content_ = std::make_shared<Edge<T>>(input);
		}
		else
		{
			for (auto& parent : this->subs_)
			{
				parent->uninitialize();
			}
			content_->set_node(input);
		}
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
	size_t type_code (void) const override
	{
		return egen::get_type<T>();
	}

	/// Implementation of iData
	std::string type_label (void) const override
	{
		return egen::name_type(egen::get_type<T>());
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
	teq::CEdgesT get_children (void) const override
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
		content_->get_node()->update();
	}

	bool is_uninit (void) const
	{
		return nullptr == content_;
	}

	std::string label_;

private:
	Placeholder (teq::ShapeSignature shape, std::string label) :
		shape_(std::vector<teq::DimT>(shape.begin(), shape.end())), label_(label) {}

	Placeholder (const Placeholder<T>& other) = default;

	Placeholder (Placeholder<T>&& other) = default;

	// teq::ShapeSignature shape_;
	teq::Shape shape_;
	// todo: replace shape with signature when functors also support incomplete shapes

	std::shared_ptr<Edge<T>> content_ = nullptr;
};

/// Placeholder's node wrapper
template <typename T>
struct PlaceholderNode final : public iNode<T>
{
	PlaceholderNode (std::shared_ptr<Placeholder<T>> phr) : phr_(phr) {}

	/// Return deep copy of this instance (with a copied variable)
	PlaceholderNode<T>* clone (void) const
	{
		return static_cast<PlaceholderNode<T>*>(clone_impl());
	}

	/// Implementation of iNode<T>
	T* data (void) override
	{
		return (T*) phr_->data();
	}

	/// Implementation of iNode<T>
	void update (void) override {}

	/// Implementation of iNode<T>
	teq::TensptrT get_tensor (void) const override
	{
		return phr_;
	}

	void assign (const NodeptrT<T>& input)
	{
		phr_->assign(input);
	}

	/// Assign Eigen tensor map to variable's internal data
	void assign (eigen::TensMapT<T>& tensmap)
	{
		phr_->assign(tensmap);
	}

	void assign (eigen::TensorT<T>& tensor)
	{
		phr_->assign(tensor);
	}

	/// Assign ShapedArr representation to variable's internal data
	void assign (teq::ShapedArr<T>& arr)
	{
		phr_->assign(arr);
	}

	/// Implementation of iNode<T>
	bool has_data (void) const override
	{
		return false == phr_->is_uninit();
	}

protected:
	iNode<T>* clone_impl (void) const override
	{
		return new PlaceholderNode(
			std::shared_ptr<Placeholder<T>>(Placeholder<T>::get(*phr_)));
	}

private:
	/// Implementation of iNode<T>
	void add_parent (Functor<T>* parent) override
	{
		phr_->subscribe(parent);
	}

	/// Implementation of iNode<T>
	void remove_parent (Functor<T>* parent) override
	{
		phr_->unsubscribe(parent);
	}

	std::shared_ptr<Placeholder<T>> phr_;
};

template <typename T>
Placeholder<T>* Placeholder<T>::get (
	teq::ShapeSignature shape, std::string label)
{
	static bool registered = register_builder<Placeholder<T>,T>(
		[](teq::TensptrT tens)
		{
			return std::make_shared<PlaceholderNode<T>>(
				std::static_pointer_cast<Placeholder<T>>(tens));
		});
	assert(registered);

	return Placeholder<T>::get(shape, label);
}

/// Smart pointer of placeholder nodes to preserve assign functions
template <typename T>
using PlaceptrT = std::shared_ptr<PlaceholderNode<T>>;

/// Return Node smart pointer of Placeholder smart pointer
template <typename T>
NodeptrT<T> convert_to_node (PlaceptrT<T> phr)
{
	return std::static_pointer_cast<iNode<T>>(phr);
}

template <typename T>
PlaceptrT<T> make_placeholder (teq::ShapeSignature shape,
	std::string label = "")
{
	return std::make_shared<PlaceholderNode<T>>(
		std::shared_ptr<Placeholder<T>>(
			Placeholder<T>::get(shape, label))
	);
}

}

#endif // ETEQ_PLACEHOLDER_HPP
