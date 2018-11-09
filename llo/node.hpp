#include <memory>
#include <function>

#include "ade/traveler.hpp"

namespace llo
{

using GradFunctorT = std::function<ade::Tensorptr(ade::ArgsT,size_t)>;

struct iDataNode
{
	virtual ~iDataNode (void) = default;

	operator ade::Tensorptr (void)
	{
		return get_tensor();
	}

	virtual OPCODE get_code (void) const = 0;

	virtual ade::Shape shape (void) const = 0;

	virtual ade::Tensorptr get_tensor (void) const = 0;
};

using NodeptrT = std::shared_ptr<iDataNode>;

using NoderefT = std::weak_ptr<iDataNode>;

using ArgsT = std::vector<std::pair<ade::CoordPtrT,NodeptrT>>;

struct Grader final : public ade::iGrader
{
	Grader (const ade::iTensor* target) : ade::iGrader(target) {}

	ade::Tensorptr chain_grad (ade::Tensorptr& wrt_child,
		ade::MappedTensor wrt_me) const override
	{
		return ade::Functor::get(make_code(MUL), {
			{ade::identity, wrt_child},
			{ade::identity, ade::Functor::get(make_code(ADD), {wrt_me})},
		});
	}

	ade::Tensorptr add_grads (ade::ArgsT& grads) const override
	{
		return ade::Functor::get(make_code(ADD), grads);
	}

	ade::Tensorptr get_grad (ade::Opcode opcode, ade::ArgsT args, size_t gradidx) const override
	{
		return gradient((OPCODE) opcode.code_, args, gradidx);
	}

	ade::Tensorptr get_scalar (const ade::Shape& shape, size_t scalar) const override
	{
		if (scalar)
		{
			return shaped_one(shape);
		}
		return shaped_zero(shape);
	}

	void set_scalar (const ade::iTensor* key, size_t scalar) override
	{
		set_grad(key, get_scalar(key->shape(), scalar));
	}

	void set_grad (const ade::iTensor* key, ade::Tensorptr value) override
	{
		grads_.emplace(key, value);
	}

	std::unordered_map<const ade::iTensor*,ade::Tensorptr> grads_;
};

struct Operation final : public iOperation
{
	Operation (GradFunctorT bwd, NoderefT node) :
		bwd_(bwd), node_(node) {}

	std::string to_string (void) const override
	{
		return name_op(node_.lock()->get_code());
	}

	size_t opnum (void) const override
	{
		return node_.lock()->get_code();
	}

	Tensorptr gradient (ArgsT args, size_t gradidx) const override
	{
		return bwd_(args, gradidx);
	}

	Tensorptr chain_grad (Tensorptr& wrt_child,
		MappedTensor wrt_me) const override
	{
		return mul(llo::get_node(wrt_child),
			copy(llo::get_node(wrt_me), wrt_me.mapper_));
	}

	Tensorptr add_grads (ArgsT& grads) const override
	{
		std::vector<llo::NodeptrT> sum_arg;
		std::transform(grads.begin(), grads.end(), std::back_inserter(sum_args),
			[](ade::MappedTensor& mtens)
			{
				return copy(llo::get_node(mtens), mtens.mapper_);
			});
		return sum(sum_arg);
	}

	NodeptrT get_node (void) const
	{
		return node_.lock();
	}

private:
	GradFunctorT bwd_;

	NoderefT node_;
};

inline NodeptrT get_node (Tensorptr& tens)
{
	return static_cast<Operation*>(wrt_child->get_opcode().get())->get_node();
}

template <OPCODE OP>
struct DataNode : public iDataNode
{
	static NodePtrT get (GradFunctorT grad, ArgsT args)
	{
		DataNode<OP>* raw = new DataNode<OP>();
		NodeptrT out(raw);
		raw->func_ = ade::Functor::get(OpPtrT(new Operation(grad, out)), args);
		return out;
	}

private:
	std::weak_ptr<ade::Functor> func_;
};

}
