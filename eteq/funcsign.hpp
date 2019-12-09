#include "eteq/link.hpp"

#ifndef ETEQ_FUNCSIGN_HPP
#define ETEQ_FUNCSIGN_HPP

namespace eteq
{

template <typename T>
struct FuncSignature final : public iLink<T>
{
	FuncSignature (egen::_GENERATED_OPCODE opcode,
		LinksT<T> args, marsh::Maps&& attrs) :
		opcode_(teq::Opcode{egen::name_op(opcode), opcode}),
		args_(args), attrs_(std::move(attrs))
	{
		if (args.empty())
		{
			logs::fatalf("cannot perform `%s` without arguments",
				egen::name_op(opcode).c_str());
		}

		marsh::iObject* shape_attr = estd::has(attrs_.contents_, eigen::shaper_key) ?
			attrs_.contents_.at(eigen::shaper_key).get() : nullptr;
		if (marsh::NumArray<double>* arr = nullptr == shape_attr ?
			nullptr : shape_attr->template cast<marsh::NumArray<double>>())
		{
			std::vector<teq::DimT> slist(
				arr->contents_.begin(), arr->contents_.end());
			shape_ = teq::ShapeSignature(slist);
		}
		else
		{
			shape_ = args_.front()->shape_sign();
		}
	}

	/// Return deep copy of this FuncSignature
	FuncSignature<T>* clone (void) const
	{
		return static_cast<FuncSignature<T>*>(clone_impl());
	}

	/// Implementation of iEdge
	teq::TensptrT get_tensor (void) const override
	{
		LinksT<T> fargs;
		fargs.reserve(args_.size());
		std::transform(args_.begin(), args_.end(), std::back_inserter(fargs),
			[](LinkptrT<T> link) { return to_link<T>(link->get_tensor()); });
		std::unique_ptr<marsh::Maps> tmp_attrs(attrs_.clone());
		return teq::TensptrT(Functor<T>::get(
			(egen::_GENERATED_OPCODE) opcode_.code_,
			fargs, std::move(*tmp_attrs)));
	}

	/// Implementation of iSignature<T>
	std::string to_string (void) const override
	{
		return opcode_.name_;
	}

	/// Implementation of iSignature<T>
	teq::ShapeSignature shape_sign (void) const override
	{
		return shape_;
	}

	/// Implementation of iEigenEdge<T>
	T* data (void) const override // violates LSP
	{
		logs::fatal("invalid call");
		return nullptr;
	}

	/// Implementation of iLink<T>
    bool has_data (void) const override // violates LSP
    {
        return false;
    }

	/// Implementation of iSignature<T>
	bool is_real (void) const override // violates LSP
	{
		return false;
	}

	teq::Opcode get_opcode (void) const
	{
		return opcode_;
	}

	teq::EdgeRefsT get_children (void) const
	{
		teq::EdgeRefsT refs;
		refs.reserve(args_.size());
		std::transform(args_.begin(), args_.end(), std::back_inserter(refs),
			[](LinkptrT<T> edge) -> const teq::iEdge&
			{
				return *edge;
			});
		return refs;
	}

	void update_child (LinkptrT<T> arg, size_t index)
	{
		if (index >= args_.size())
		{
			logs::fatalf("cannot modify argument %d "
				"when there are only %d arguments",
				index, args_.size());
		}
        static_cast<iLink<T>*>(args_[index].get())->unsubscribe(this);
        args_[index] = arg;
        arg->subscribe(this);
	}

private:
    iLink<T>* clone_impl (void) const override
    {
		std::unique_ptr<marsh::Maps> tmp_attrs(attrs_.clone());
        return new FuncSignature((egen::_GENERATED_OPCODE) opcode_.code_,
            args_, std::move(*tmp_attrs));
    }

	void subscribe (Functor<T>* parent) override {}

	void unsubscribe (Functor<T>* parent) override {}

	/// Operation encoding
	teq::Opcode opcode_;

	/// Shape info built at construction time according to arguments
	teq::ShapeSignature shape_;

	/// Tensor arguments (and children)
	LinksT<T> args_;

	marsh::Maps attrs_;
};

}

#endif // ETEQ_FUNCSIGN_HPP
