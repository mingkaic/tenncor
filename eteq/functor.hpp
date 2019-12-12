//
/// functor.hpp
/// eteq
///
/// Purpose:
/// Eigen functor implementation of operable func
///

#include "teq/iopfunc.hpp"

#include "tag/locator.hpp"

#include "eigen/generated/opcode.hpp"
#include "eigen/packattr.hpp"

#include "eteq/link.hpp"
#include "eteq/observable.hpp"
#include "eteq/funcsign.hpp"

#ifndef ETEQ_FUNCTOR_HPP
#define ETEQ_FUNCTOR_HPP

namespace eteq
{

#define CHOOSE_PARSER(OPCODE)\
outshape = ShapeParser<OPCODE>().shape(attrs, shapes);

/// Functor implementation of operable functor of Eigen operators
template <typename T>
struct Functor final : public teq::iOperableFunc, public Observable<Functor<T>*>
{
	/// Return Functor given opcodes mapped to Eigen operators in operator.hpp
	/// Return nullptr if functor is redundant
	static teq::TensptrT get (egen::_GENERATED_OPCODE opcode,
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
			return links.front()->get_tensor();
		}
		return teq::TensptrT(new Functor<T>(opcode,
			teq::Shape(*outshape), links, std::move(attrs)));
	}

	~Functor (void)
	{
		for (LinkptrT<T> arg : links_)
		{
			arg->unsubscribe(this);
		}
	}

	/// Return deep copy of this Functor
	Functor<T>* clone (void) const
	{
		return static_cast<Functor<T>*>(clone_impl());
	}

	Functor (Functor<T>&& other) = delete;

	Functor<T>& operator = (const Functor<T>& other) = delete;

	Functor<T>& operator = (Functor<T>&& other) = delete;

	/// Implementation of iTensor
	void accept (teq::iTraveler& visiter) override
	{
		visiter.visit(this);
	}

	/// Implementation of iTensor
	teq::Shape shape (void) const override
	{
		return shape_;
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return opcode_.name_;
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
		auto link = static_cast<iLink<T>*>(links_[index].get());
		if (arg != link->get_tensor())
		{
			uninitialize();
			link->unsubscribe(this);
			teq::Shape nexshape = arg->shape();
			teq::Shape curshape = link->shape();
			if (false == nexshape.compatible_after(curshape, 0))
			{
				logs::fatalf("cannot update child %d to argument with "
					"incompatible shape %s (requires shape %s)",
					index, nexshape.to_string().c_str(),
					curshape.to_string().c_str());
			}
			links_[index] = to_link<T>(arg);
			links_[index]->subscribe(this);
		}
	}

	/// Implementation of iOperableFunc
	void update (void) override
	{
		if (false == has_data())
		{
			initialize();
		}
		out_->assign();
	}

	/// Implementation of iData
	void* data (void) override
	{
		if (false == has_data())
		{
			initialize();
		}
		return out_->get_ptr();
	}

	/// Implementation of iData
	const void* data (void) const override
	{
		if (false == has_data())
		{
			logs::fatal("cannot get data of uninitialized functor");
		}
		return out_->get_ptr();
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
		return sizeof(T) * shape().n_elems();
	}

	bool has_data (void) const
	{
		return nullptr != out_;
	}

	/// Removes internal Eigen data object
	void uninitialize (void)
	{
		if (has_data())
		{
			for (auto& parent : this->subs_)
			{
				parent->uninitialize();
			}
			out_ = nullptr;
		}
	}

	/// Populate internal Eigen data object
	void initialize (void)
	{
		eigen::EEdgeRefsT<T> refs;
		refs.reserve(links_.size());
		std::transform(links_.begin(), links_.end(), std::back_inserter(refs),
			[](LinkptrT<T> edge) -> const eigen::iEigenEdge<T>&
			{
				return *edge;
			});
		egen::typed_exec<T>((egen::_GENERATED_OPCODE) opcode_.code_,
			out_, shape_, refs, *this);
	}

private:
	Functor (egen::_GENERATED_OPCODE opcode, teq::Shape shape,
		LinksT<T> links, marsh::Maps&& attrs) :
		opcode_(teq::Opcode{egen::name_op(opcode), opcode}),
		shape_(shape), links_(links), attrs_(std::move(attrs))
	{
		common_init();
	}

	Functor (const Functor<T>& other) :
		opcode_(other.opcode_),
		shape_(other.shape_),
		links_(other.links_)
	{
		std::unique_ptr<marsh::Maps> mattr(other.attrs_.clone());
		attrs_ = std::move(*mattr);
		common_init();
	}

	teq::iTensor* clone_impl (void) const override
	{
		return new Functor<T>(*this);
	}

	void common_init (void)
	{
		for (LinkptrT<T> arg : links_)
		{
			arg->subscribe(this);
		}
#ifndef SKIP_INIT
		if (std::all_of(links_.begin(), links_.end(),
			[](LinkptrT<T>& link)
			{
				return link->has_data();
			}))
		{
			initialize();
		}
#endif // SKIP_INIT
	}

	eigen::EigenptrT<T> out_ = nullptr;

	/// Operation encoding
	teq::Opcode opcode_;

	/// Shape info built at construction time according to arguments
	teq::Shape shape_;

	/// Tensor arguments (and children)
	LinksT<T> links_;

	marsh::Maps attrs_;
};

#undef CHOOSE_PARSER

/// Functor's node wrapper
template <typename T>
struct FuncLink final : public iLink<T>
{
	FuncLink (std::shared_ptr<Functor<T>> func) : func_(func)
	{
		if (func == nullptr)
		{
			logs::fatal("cannot link a null func");
		}
	}

	/// Return deep copy of this instance (with a copied functor)
	FuncLink<T>* clone (void) const
	{
		return static_cast<FuncLink<T>*>(clone_impl());
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
	T* data (void) const override
	{
		return (T*) func_->data();
	}

	/// Implementation of iLink<T>
	bool has_data (void) const override
	{
		return func_->has_data();
	}

	/// Implementation of iLink<T>
	teq::TensptrT get_tensor (void) const override
	{
		return func_;
	}

	/// Implementation of iSignature
	teq::TensptrT build_tensor (void) const override
	{
		return func_;
	}

	/// Implementation of iSignature
	teq::ShapeSignature shape_sign (void) const override
	{
		teq::Shape shape = func_->shape();
		return teq::ShapeSignature(
			std::vector<teq::DimT>(shape.begin(), shape.end()));
	}

private:
	FuncLink (const FuncLink<T>& other) = default;

	iLink<T>* clone_impl (void) const override
	{
		return new FuncLink(std::shared_ptr<Functor<T>>(func_->clone()));
	}

	/// Implementation of iLink<T>
	void subscribe (Functor<T>* parent) override
	{
		func_->subscribe(parent);
	}

	/// Implementation of iLink<T>
	void unsubscribe (Functor<T>* parent) override
	{
		func_->unsubscribe(parent);
	}

	std::shared_ptr<Functor<T>> func_;
};

}

#endif // ETEQ_FUNCTOR_HPP
