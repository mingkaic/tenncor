#include "eteq/link.hpp"
#include "eteq/shaper.hpp"

#ifndef ETEQ_FUNCSIGN_HPP
#define ETEQ_FUNCSIGN_HPP

namespace eteq
{

#define CHOOSE_PARSER(OPCODE)\
outshape = ShapeParser<OPCODE>().shape(attrs, shapes);

template <typename T>
struct FuncSignature final : public iLink<T>
{
	static LinkptrT<T> get (egen::_GENERATED_OPCODE opcode,
		LinksT<T> links, marsh::Maps&& attrs)
	{
		if (links.empty())
		{
			logs::fatalf("cannot perform `%s` without arguments",
				egen::name_op(opcode).c_str());
		}

		ShapesT shapes;
		shapes.reserve(links.size());
		std::transform(links.begin(), links.end(), std::back_inserter(shapes),
			[](LinkptrT<T> link)
			{
				return link->shape_sign();
			});

		ShapeOpt outshape;
		OPCODE_LOOKUP(CHOOSE_PARSER, opcode)
		if (false == outshape.has_value())
		{
			return links.front();
		}
		return LinkptrT<T>(new FuncSignature<T>(
			opcode, *outshape, links, std::move(attrs)));
	}

	/// Return deep copy of this FuncSignature
	FuncSignature<T>* clone (void) const
	{
		return static_cast<FuncSignature<T>*>(clone_impl());
	}

	/// Implementation of iAttributed
	std::vector<std::string> ls_attrs (void) const override
	{
		return attrs_.ls_attrs();
	}

	/// Implementation of iAttributed
	const marsh::iObject* get_attr (std::string attr_name) const override
	{
		return attrs_.get_attr(attr_name);
	}

	/// Implementation of iAttributed
	void add_attr (std::string attr_key, marsh::ObjptrT&& attr_val) override
	{
		attrs_.add_attr(attr_key, std::move(attr_val));
	}

	/// Implementation of iAttributed
	void rm_attr (std::string attr_key) override
	{
		attrs_.rm_attr(attr_key);
	}

	/// Implementation of iEdge
	teq::TensptrT get_tensor (void) const override
	{
		LinksT<T> fargs;
		fargs.reserve(links_.size());
		std::transform(links_.begin(), links_.end(), std::back_inserter(fargs),
			[](LinkptrT<T> link) { return to_link<T>(link->get_tensor()); });
		std::unique_ptr<marsh::Maps> tmp_attrs(attrs_.clone());
		return Functor<T>::get((egen::_GENERATED_OPCODE) opcode_.code_,
			fargs, std::move(*tmp_attrs));
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
		refs.reserve(links_.size());
		std::transform(links_.begin(), links_.end(), std::back_inserter(refs),
			[](LinkptrT<T> edge) -> const teq::iEdge&
			{
				return *edge;
			});
		return refs;
	}

	void update_child (LinkptrT<T> arg, size_t index)
	{
		if (index >= links_.size())
		{
			logs::fatalf("cannot modify argument %d "
				"when there are only %d arguments",
				index, links_.size());
		}
		static_cast<iLink<T>*>(links_[index].get())->unsubscribe(this);
		links_[index] = arg;
		arg->subscribe(this);
	}

private:
	FuncSignature (egen::_GENERATED_OPCODE opcode,
		teq::ShapeSignature shape, LinksT<T> links, marsh::Maps&& attrs) :
		opcode_(teq::Opcode{egen::name_op(opcode), opcode}),
		shape_(shape), links_(links), attrs_(std::move(attrs)) {}

	FuncSignature (const FuncSignature<T>& other) :
		opcode_(other.opcode_),
		shape_(other.shape_),
		links_(other.links_)
	{
		std::unique_ptr<marsh::Maps> mattr(other.attrs_.clone());
		attrs_ = std::move(*mattr);
	}

	iLink<T>* clone_impl (void) const override
	{
		return new FuncSignature(*this);
	}

	void subscribe (Functor<T>* parent) override {}

	void unsubscribe (Functor<T>* parent) override {}

	/// Operation encoding
	teq::Opcode opcode_;

	/// Shape info built at construction time according to arguments
	teq::ShapeSignature shape_;

	/// Tensor arguments (and children)
	LinksT<T> links_;

	marsh::Maps attrs_;
};

#undef CHOOSE_PARSER

}

#endif // ETEQ_FUNCSIGN_HPP
