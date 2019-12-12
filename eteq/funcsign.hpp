#include "eteq/link.hpp"
#include "eteq/shaper.hpp"

#ifndef ETEQ_FUNCSIGN_HPP
#define ETEQ_FUNCSIGN_HPP

namespace eteq
{

template <typename T>
struct FuncSignature final : public teq::iFunctor, public iSignature
{
	static LinkptrT<T> get (egen::_GENERATED_OPCODE opcode,
		LinksT<T> links, marsh::Maps&& attrs);

	/// Return deep copy of this FuncSignature
	FuncSignature<T>* clone (void) const
	{
		return static_cast<FuncSignature<T>*>(clone_impl());
	}

	FuncSignature (FuncSignature<T>&& other) = delete;

	FuncSignature<T>& operator = (const FuncSignature<T>& other) = delete;

	FuncSignature<T>& operator = (FuncSignature<T>&& other) = delete;

	/// Implementation of iTensor
	void accept (teq::iTraveler& visiter) override
	{
		visiter.visit(this);
	}

	/// Implementation of iTensor
	teq::Shape shape (void) const override
	{
		return build_tensor()->shape();
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return opcode_.name_;
	}

	/// Implementation of iAttributed
	const marsh::iObject* get_attr (std::string attr_key) const override
	{
		return attrs_.get_attr(attr_key);
	}

	/// Implementation of iAttributed
	std::vector<std::string> ls_attrs (void) const override
	{
		return attrs_.ls_attrs();
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

	/// Implementation of iFunctor
	teq::Opcode get_opcode (void) const override
	{
		return opcode_;
	}

	/// Implementation of iFunctor
	teq::TensptrsT get_children (void) const override
	{
		teq::TensptrsT refs;
		refs.reserve(links_.size());
		std::transform(links_.begin(), links_.end(), std::back_inserter(refs),
			[](LinkptrT<T> edge)
			{
				return edge->get_tensor();
			});
		return refs;
	}

	/// Implementation of iFunctor
	void update_child (teq::TensptrT arg, size_t index) override
	{
		if (index >= links_.size())
		{
			logs::fatalf("cannot modify argument %d "
				"when there are only %d arguments",
				index, links_.size());
		}
		links_[index] = to_link<T>(arg);
	}

	/// Implementation of iSignature
	teq::TensptrT build_tensor (void) const override
	{
		teq::TensptrsT args;
		args.reserve(links_.size());
		std::transform(links_.begin(), links_.end(), std::back_inserter(args),
			[](LinkptrT<T> link) { return link->build_tensor(); });
		if (std::any_of(args.begin(), args.end(),
			[](teq::TensptrT arg) { return nullptr == arg; }))
		{
			return nullptr;
		}

		LinksT<T> flinks;
		flinks.reserve(args.size());
		std::transform(args.begin(), args.end(), std::back_inserter(flinks),
			[](teq::TensptrT arg) { return to_link<T>(arg); });
		std::unique_ptr<marsh::Maps> tmp_attrs(attrs_.clone());
		return Functor<T>::get((egen::_GENERATED_OPCODE) opcode_.code_,
			flinks, std::move(*tmp_attrs));
	}

	/// Implementation of iSignature
	teq::ShapeSignature shape_sign (void) const override
	{
		return shape_;
	}

private:
	FuncSignature (egen::_GENERATED_OPCODE opcode, teq::ShapeSignature shape,
		LinksT<T> links, marsh::Maps&& attrs) :
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

	/// Implementation of iTensor
	teq::iTensor* clone_impl (void) const override
	{
		return new FuncSignature<T>(*this);
	}

	/// Operation encoding
	teq::Opcode opcode_;

	/// Shape info built at construction time according to arguments
	teq::ShapeSignature shape_;

	/// Tensor arguments (and children)
	LinksT<T> links_;

	marsh::Maps attrs_;
};

/// FuncSignLink is a builder of Functor<T> given opcode, link, and attrs,
/// Since it's a builder, it maintains no ownership of the functors it builds
template <typename T>
struct FuncSignLink final : public iLink<T>
{
	FuncSignLink (std::shared_ptr<FuncSignature<T>> func) : func_(func)
	{
		if (func == nullptr)
		{
			logs::fatal("cannot link a null func");
		}
	}

	/// Return deep copy of this FuncSignLink
	FuncSignLink<T>* clone (void) const
	{
		return static_cast<FuncSignLink<T>*>(clone_impl());
	}

	/// Implementation of iAttributed
	std::vector<std::string> ls_attrs (void) const override
	{
		return func_->ls_attrs();
	}

	/// Implementation of iAttributed
	const marsh::iObject* get_attr (std::string attr_key) const override
	{
		return func_->get_attr(attr_key);
	}

	/// Implementation of iAttributed
	void add_attr (std::string attr_key, marsh::ObjptrT&& attr_val) override
	{
		func_->add_attr(attr_key, std::move(attr_val));
	}

	/// Implementation of iAttributed
	void rm_attr (std::string attr_key) override
	{
		func_->rm_attr(attr_key);
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

	/// Implementation of iLink<T>
	teq::TensptrT get_tensor (void) const override
	{
		return func_;
	}

	/// Implementation of iSignature
	teq::TensptrT build_tensor (void) const override
	{
		return func_->build_tensor();
	}

	/// Implementation of iSignature
	teq::ShapeSignature shape_sign (void) const override
	{
		return func_->shape_sign();
	}

private:
	iLink<T>* clone_impl (void) const override
	{
		return new FuncSignLink(*this);
	}

	void subscribe (Functor<T>* parent) override {}

	void unsubscribe (Functor<T>* parent) override {}

	std::shared_ptr<FuncSignature<T>> func_;
};

#define CHOOSE_PARSER(OPCODE)\
outshape = ShapeParser<OPCODE>().shape(attrs, shapes);

template <typename T>
LinkptrT<T> FuncSignature<T>::get (egen::_GENERATED_OPCODE opcode,
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
	return std::make_shared<FuncSignLink<T>>(
		std::shared_ptr<FuncSignature<T>>(new FuncSignature<T>(opcode,
			*outshape, links, std::move(attrs))));
}

#undef CHOOSE_PARSER

}

#endif // ETEQ_FUNCSIGN_HPP
