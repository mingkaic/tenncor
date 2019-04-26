#include "ade/ifunctor.hpp"

#ifndef EAD_REPRESENT_HPP
#define EAD_REPRESENT_HPP

namespace ead
{

namespace opt
{

static bool is_commutative (size_t opcode)
{
	switch (opcode)
	{
		case ADD:
		case MUL:
		case MIN:
		case MAX:
			return true;
		default:
			break;
	}
	return false;
}

struct RuleContext;

struct iReprNode;

using RepptrT = std::shared_ptr<iReprNode>;

struct iRuleNode;

using RuleptrT = std::shared_ptr<iRuleNode>;

struct ConstRep;

struct LeafRep;

struct FuncRep;

struct iReprNode
{
	virtual ~iReprNode (void) = default;

	virtual bool rulify (RuleContext& ctx, const RuleptrT& rule) = 0;

	virtual std::string get_identifier (void) const = 0;

	virtual bool is_constant (void) const = 0;
};

struct ReprArg final
{
	ReprArg (RepptrT arg,
		ade::CoordptrT shaper, ade::CoordptrT coorder) :
		arg_(arg), shaper_(shaper), coorder_(coorder)
	{
		if (nullptr == arg)
		{
			logs::fatal("created a representation argument with null argument");
		}
	}

	RepptrT arg_;

	ade::CoordptrT shaper_;

	ade::CoordptrT coorder_;
};

using RepArgsT = std::vector<ReprArg>;

struct iRuleNode
{
	virtual ~iRuleNode (void) = default;

	virtual bool process (RuleContext& ctx, ConstRep* leaf) const = 0;

	virtual bool process (RuleContext& ctx, LeafRep* leaf) const = 0;

	virtual bool process (RuleContext& ctx, FuncRep* func) const = 0;

	virtual size_t get_height (void) const = 0;
};

struct RuleArg final
{
	RuleArg (RuleptrT arg,
		ade::CoordptrT shaper, ade::CoordptrT coorder) :
		arg_(arg), shaper_(shaper), coorder_(coorder)
	{
		if (nullptr == arg)
		{
			logs::fatal("created a rule argument with null argument");
		}
	}

	RuleptrT arg_;

	ade::CoordptrT shaper_;

	ade::CoordptrT coorder_;
};

using RuleArgsT = std::vector<RuleArg>;

template <typename T>
struct ConstRep final : public iReprNode
{
	ConstRep (T* data, ade::Shape shape) :
		data_(data, data + shape_.n_elems()), shape_(shape)
	{
		std::string data_str;
		size_t n = shape.n_elems();
		if (std::all_of(data + 1, data + n,
			[&](const T& e) { return e == data[0]; }))
		{
			// is a scalar
			std::stringstream ss;
			ss << "scalar_" << std::hex << data[0];
			data_str = ss.str();
		}
		else
		{
			// todo: make encoding better so we don't lose precision
			data_str = "array_" + fmts::join("", data, data + n);
		}
		identifier_ = fmts::join("", shape.begin(), shape.end()) + data_str;
	}

	bool rulify (RuleContext& ctx, const RuleptrT& rule) override
	{
		return rule.process(ctx, this);
	}

	std::string get_identifier (void) const override
	{
		return identifier_;
	}

	bool is_constant (void) const override
	{
		return true;
	}

	std::vector<T> data_;

	ade::Shape shape_;

	std::string identifier_;
};

template <typename T>
struct LeafRep final : public iReprNode
{
	LeafRep (ade::iLeaf* leaf) :
		leaf_(leaf), identifier_(pid(leaf)) {}

	bool rulify (RuleContext& ctx, const RuleptrT& rule) override
	{
		return rule.process(ctx, this);
	}

	std::string get_identifier (void) const override
	{
		return identifier_;
	}

	bool is_constant (void) const override
	{
		return false;
	}

	ade::iLeaf* leaf_;

	std::string identifier_;
};

template <typename T>
struct FuncRep final : public Representation
{
	FuncRep (ade::Opcode op, RepArgsT args) :
		identifier_(pid(leaf)),
		op_(op), args_(args) {}

	bool rulify (RuleContext& ctx, const RuleptrT& rule) override
	{
		return rule.process(ctx, this);
	}

	std::string get_identifier (void) const override
	{
		return identifier_;
	}

	bool is_constant (void) const override
	{
		return std::all_of(args_.begin(), args_.end(),
			[](FuncArg& arg)
			{
				return arg.arg_->is_constant();
			});
	}

	std::string get_imprint (void) const
	{
		std::vector<std::string> arg_ids;
		arg_ids.reserve(args_.size());
		for (auto& arg : args_)
		{
			std::string id = arg.get_identifier();
			auto& shaper = arg.shaper_;
			auto& coorder = arg.coorder_;
			if (nullptr != shaper)
			{
				// todo: make shaper encoding shorter
				id += shaper->to_string();
			}
			id += "_";
			if (nullptr != coorder)
			{
				// todo: make coorder encoding shorter
				id += coorder->to_string();
			}
			arg_ids.push_back(id);
		}
		if (is_commutative(op.code_))
		{
			std::sort(arg_ids.begin(), arg_ids.end());
		}
		return op_.name_ + ":" +
			fmts::join(",", arg_ids.begin(), arg_ids.end());
	}

	std::string identifier_;

	ade::Opcode op_;

	RepArgsT args_;
};

template <typename T>
using FuncRepptrT = std::shared_ptr<FuncRep<T>>;

template <typename T>
struct Representer final : public ade::iTraveler
{
	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		if (reps_.end() == reps_.find(leaf))
		{
			if (static_cast<iLeaf<T>*>(leaf)->is_const())
			{
				reps_.emplace(leaf, std::make_shared<ConstRep>(
					(T*) leaf->data(), leaf->shape()));
			}
			else
			{
				reps_.emplace(leaf, std::make_shared<LeafRep>(leaf));
			}
		}
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (reps_.end() == reps_.find(func))
		{
			auto& children = func->get_children();
			std::vector<FuncRep::FuncArg> args;
			args.reserve(children.size());
			for (auto& child : children)
			{
				auto tens = child.get_tensor();
				tens->accept(*this);
				args.push_back(FuncRep::FuncArg{
					reps_[tens.get()],
					child.get_shaper(),
					child.get_coorder(),
				})
			}
			auto func_rep = std::make_shared<FuncRep>(
				func->get_opcode(), args);
			register_func(func_rep);
		}
	}

	void register_func (FuncRepptrT func_rep)
	{
		// precalculate constants (todo: make this optional?)
		if (func_rep->is_constant())
		{
			auto f = static_cast<Functor<T>*>(func);
			f->update();
			reps_.emplace(func, std::make_shared<ConstRep>(
				f->data(), f->shape()))
		}
		else
		{
			reps_.emplace(func, func_rep);

			// reuse node
			std::string arg_imprint = func_rep->get_imprint();
			auto it = imprints_.find(arg_imprint);
			if (imprints_.end() == it)
			{
				imprints_.emplace(arg_imprint,
					func->get_identifier());
			}
			else
			{
				func_rep->identifier_ = it->second;
			}

			for (size_t i = 0, n = func_rep->args_.size(); i < n; ++i)
			{
				auto& arg = func_rep->args_[i];
				parents_[arg.arg_].push_back({func_rep, i});
			}
		}
	}

	std::unordered_map<ade::iTensor*,RepptrT> reps_;

	// 1-to-1 map longer function imprint to a shorter id
	std::unordered_map<std::string,std::string> imprints_;

	std::unordered_map<RepptrT,std::vector<
		std::pair<FuncRepptrT,size_t>>> parents_;
};

// todo: make this a visitor to avoid casting
template <typename T>
void unravel (std::unordered_map<RepptrT,NodeptrT<T>>& unravelled,
	const RepptrT<T>& rep, const std::unordered_map<iTensor*,TensrefT>& owner)
{
	NodeptrT<T> out;
	if (auto f = dynamic_cast<FuncRep<T>*>(rep.get()))
	{
		ArgsT<T> args;
		args.reserve(f->args_.size());
		for (auto& arg : f->args_)
		{
			unravel(unravelled, arg.arg_, owner);
			CoordptrT coorder;
			if (arg.coorder_)
			{
				coorder = std::static_pointer_cast<CoordMap>(arg.coorder_);
			}
			args.push_back(FuncArg<T>(unravelled[arg.arg_],
				arg.shaper_, coorder));
		}
		out = std::shared_ptr<Functor<T>>(Functor<T>::get(f->op_, args));
	}
	else if (auto cst = dynamic_cast<ConstRep<T>*>(rep.get()))
	{
		out = make_constant(cst->data_, cst->shape_);
	}
	else
	{
		auto lef = static_cast<LeafRep<T>*>(rep.get());
		out = to_node<T>(owner[lef.leaf_]);
	}
	unravelled.emplace(rep, out);
}

template <typename T>
NodesT<T> unrepresent (Representer<T>& repr, NodesT<T> roots)
{
	NodesT<T> out;
	out.reserve(roots.size());
	std::unordered_map<RepptrT,NodeptrT<T>> unravelled;
	for (auto& root : roots)
	{
		auto tens = root->get_tensor();
		auto owners = ade::track_owners(tens);
		auto& rep = repr.reps_[tens];
		unravel(unravelled, rep, owners);
		out.push_back(unravelled[rep]);
	}
	return roots;
}

}

}

#endif // EAD_REPRESENT_HPP
