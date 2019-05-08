#include <random>

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
		case age::ADD:
		case age::MUL:
		case age::MIN:
		case age::MAX:
			return true;
		default:
			break;
	}
	return false;
}

template <typename T>
struct RuleContext;

template <typename T>
struct iReprNode;

template <typename T>
using RepptrT = std::shared_ptr<iReprNode<T>>;

template <typename T>
struct iRuleNode;

template <typename T>
using RuleptrT = std::shared_ptr<iRuleNode<T>>;

template <typename T>
struct ConstRep;

template <typename T>
struct LeafRep;

template <typename T>
struct FuncRep;

template <typename T, typename = typename std::enable_if<
	std::is_arithmetic<T>::value, T>::type>
struct NumRange
{
	NumRange (T bound1, T bound2) :
		lower_(std::min(bound1, bound2)),
		upper_(std::max(bound1, bound2)) {}

	T lower_;

	T upper_;
};

template <typename T>
struct iReprNode
{
	virtual ~iReprNode (void) = default;

	virtual bool rulify (RuleContext<T>& ctx, const RuleptrT<T>& rule) = 0;

	virtual std::string get_identifier (void) const = 0;

	virtual bool is_constant (void) const = 0;

	virtual NumRange<size_t> get_heights (void) const = 0;
};

template <typename T>
struct ReprArg final
{
	ReprArg (RepptrT<T> arg,
		ade::CoordptrT shaper, ade::CoordptrT coorder) :
		arg_(arg), shaper_(shaper), coorder_(coorder)
	{
		if (nullptr == arg)
		{
			logs::fatal("created a representation argument with null argument");
		}
	}

	RepptrT<T> arg_;

	ade::CoordptrT shaper_;

	ade::CoordptrT coorder_;
};

template <typename T>
using RepArgsT = std::vector<ReprArg<T>>;

template <typename T>
struct iRuleNode
{
	virtual ~iRuleNode (void) = default;

	virtual bool process (RuleContext<T>& ctx, ConstRep<T>* leaf) const = 0;

	virtual bool process (RuleContext<T>& ctx, LeafRep<T>* leaf) const = 0;

	virtual bool process (RuleContext<T>& ctx, FuncRep<T>* func) const = 0;

	virtual size_t get_minheight (void) const = 0;
};

template <typename T>
struct RuleArg final
{
	RuleArg (RuleptrT<T> arg,
		ade::CoordptrT shaper, ade::CoordptrT coorder) :
		arg_(arg), shaper_(shaper), coorder_(coorder)
	{
		if (nullptr == arg)
		{
			logs::fatal("created a rule argument with null argument");
		}
	}

	RuleptrT<T> arg_;

	ade::CoordptrT shaper_;

	ade::CoordptrT coorder_;
};

template <typename T>
using RuleArgsT = std::vector<RuleArg<T>>;

template <typename T>
struct ConstRep final : public iReprNode<T>
{
	ConstRep (T* data, ade::Shape shape) :
		data_(data, data + shape.n_elems()), shape_(shape)
	{
		std::string data_str;
		size_t n = shape.n_elems();
		if (std::all_of(data + 1, data + n,
			[&](const T& e) { return e == data[0]; }))
		{
			// is a scalar
			std::stringstream ss;
			double scalar = data[0];
			ss << "_scalar_" << fmts::to_string(scalar);
			data_str = ss.str();
		}
		else
		{
			// todo: make encoding better so we don't lose precision
			std::vector<double> ddata(data, data + n);
			data_str = "_array_" + fmts::join("", ddata.begin(), ddata.end());
		}
		identifier_ = fmts::join("", shape.begin(), shape.end()) + data_str;
	}

	bool rulify (RuleContext<T>& ctx, const RuleptrT<T>& rule) override
	{
		return rule->process(ctx, this);
	}

	std::string get_identifier (void) const override
	{
		return identifier_;
	}

	bool is_constant (void) const override
	{
		return true;
	}

	NumRange<size_t> get_heights (void) const override
	{
		return NumRange<size_t>{1, 1};
	}

	std::vector<T> data_;

	ade::Shape shape_;

private:
	std::string identifier_;
};

template <typename T>
struct LeafRep final : public iReprNode<T>
{
	LeafRep (ade::iLeaf* leaf, size_t id) :
		leaf_(leaf), identifier_(id) {}

	bool rulify (RuleContext<T>& ctx, const RuleptrT<T>& rule) override
	{
		return rule->process(ctx, this);
	}

	std::string get_identifier (void) const override
	{
		return fmts::sprintf("var_%d", identifier_);
	}

	bool is_constant (void) const override
	{
		return false;
	}

	NumRange<size_t> get_heights (void) const override
	{
		return NumRange<size_t>{1, 1};
	}

	ade::iLeaf* leaf_;

private:
	size_t identifier_;
};

using ImprintMapT = std::unordered_map<std::string,std::string>;

template <typename T>
struct FuncRep final : public iReprNode<T>
{
	FuncRep (ade::Opcode op, RepArgsT<T> args) :
		op_(op), args_(args) {}

	bool rulify (RuleContext<T>& ctx, const RuleptrT<T>& rule) override
	{
		return rule->process(ctx, this);
	}

	std::string get_identifier (void) const override
	{
		std::vector<std::string> arg_ids;
		arg_ids.reserve(args_.size());
		for (auto& arg : args_)
		{
			std::string id = arg.arg_->get_identifier();
			auto& shaper = arg.shaper_;
			auto& coorder = arg.coorder_;
			if (false == ade::is_identity(shaper.get()))
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
		if (is_commutative(op_.code_))
		{
			std::sort(arg_ids.begin(), arg_ids.end());
		}
		return op_.name_ + "(" +
			fmts::join(",", arg_ids.begin(), arg_ids.end()) + ")";
	}

	bool is_constant (void) const override
	{
		return std::all_of(args_.begin(), args_.end(),
			[](const ReprArg<T>& arg)
			{
				return arg.arg_->is_constant();
			});
	}

	NumRange<size_t> get_heights (void) const override
	{
		size_t min = std::numeric_limits<size_t>::max();
		size_t max = 0;
		for (const ReprArg<T>& arg : args_)
		{
			auto ranges = arg.arg_->get_heights();
			if (min > ranges.lower_)
			{
				min = ranges.lower_;
			}
			if (max < ranges.upper_)
			{
				max = ranges.upper_;
			}
		}
		return NumRange<size_t>{min + 1, max + 1};
	}

	const RepArgsT<T>& get_args (void) const
	{
		return args_;
	}

	void set_arg (size_t arg_idx, RepptrT<T>& arg)
	{
		args_[arg_idx].arg_ = arg;
	}

	ade::Opcode op_;

private:
	RepArgsT<T> args_;
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
				reps_.emplace(leaf, std::make_shared<ConstRep<T>>(
					(T*) leaf->data(), leaf->shape()));
			}
			else
			{
				reps_.emplace(leaf,
					std::make_shared<LeafRep<T>>(leaf, reps_.size()));
			}
		}
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (reps_.end() == reps_.find(func))
		{
			auto& children = func->get_children();
			RepArgsT<T> args;
			args.reserve(children.size());
			for (auto& child : children)
			{
				auto tens = child.get_tensor();
				tens->accept(*this);
				args.push_back(ReprArg<T>{
					reps_[tens.get()],
					child.get_shaper(),
					child.get_coorder(),
				});
			}
			// precalculate constants (todo: make this optional?)
			if (std::all_of(args.begin(), args.end(),
				[](const ReprArg<T>& arg)
				{
					return arg.arg_->is_constant();
				}))
			{
				auto f = static_cast<Functor<T>*>(func);
				f->update();
				reps_[func] = std::make_shared<ConstRep<T>>(
					f->data(), f->shape());
			}
			else
			{
				reps_[func] = make_func(func->get_opcode(), args);
			}
		}
	}

	FuncRepptrT<T> make_func (ade::Opcode opcode, const RepArgsT<T>& args)
	{
		auto func_rep = std::make_shared<FuncRep<T>>(opcode, args);
		for (size_t i = 0, n = args.size(); i < n; ++i)
		{
			auto& arg = args[i];
			parents_[arg.arg_].push_back({func_rep, i});
		}
		return func_rep;
	}

	void replace_rep (ade::iTensor* orig, RepptrT<T> repl_rep)
	{
		auto it = reps_.find(orig);
		if (reps_.end() == it || it->second == repl_rep)
		{
			return;
		}
		auto& orig_rep = it->second;
		auto& repl_parents = parents_[repl_rep];
		auto& orig_parents = parents_[orig_rep];
		for (auto& orig_parent : orig_parents)
		{
			FuncRepptrT<T>& parent = orig_parent.first;
			size_t arg_idx = orig_parent.second;
			parent->set_arg(arg_idx, repl_rep);
			repl_parents.push_back({parent, arg_idx});
		}
		reps_[orig] = repl_rep;
	}

	std::unordered_map<ade::iTensor*,RepptrT<T>> reps_;

	std::unordered_map<RepptrT<T>,std::vector<
		std::pair<FuncRepptrT<T>,size_t>>> parents_;
};

// todo: make this a visitor to avoid casting
template <typename T>
void unravel (std::unordered_map<RepptrT<T>,NodeptrT<T>>& unravelled,
	const RepptrT<T>& rep, const ade::OwnerMapT& owners)
{
	if (unravelled.end() != unravelled.find(rep))
	{
		return;
	}
	NodeptrT<T> out;
	if (auto f = dynamic_cast<FuncRep<T>*>(rep.get()))
	{
		auto& func_args = f->get_args();
		size_t nargs = func_args.size();
		ArgsT<T> args;
		args.reserve(nargs);
		for (auto& arg : func_args)
		{
			unravel(unravelled, arg.arg_, owners);
			CoordptrT coorder;
			if (arg.coorder_)
			{
				coorder = std::static_pointer_cast<CoordMap>(arg.coorder_);
			}
			args.push_back(FuncArg<T>(unravelled[arg.arg_],
				arg.shaper_, coorder));
		}
		if (is_commutative(f->op_.code_) && nargs == 1)
		{
			// todo: make this check somewhere else
			assert(ade::is_identity(args[0].get_shaper().get()));
			assert(nullptr == args[0].get_coorder());
			out = args[0].get_node();
		}
		else if (nargs > 2)
		{
			FuncArg<T> left = args[0];
			FuncArg<T> right = args[1];
			for (size_t i = 2; i < nargs; ++i)
			{
				left = FuncArg<T>{
					make_functor<T>(f->op_, {left, right}),
					ade::identity, CoordptrT()
				};
				right = args[i];
			}
			out = make_functor<T>(f->op_, {left, right});
		}
		else
		{
			out = make_functor(f->op_, args);
		}
	}
	else if (auto cst = dynamic_cast<ConstRep<T>*>(rep.get()))
	{
		out = make_constant(cst->data_.data(), cst->shape_);
	}
	else
	{
		auto lef = static_cast<LeafRep<T>*>(rep.get());
		out = to_node<T>(owners.at(lef->leaf_).lock());
	}
	unravelled.emplace(rep, out);
}

template <typename T>
NodesT<T> unrepresent (Representer<T>& repr, const NodesT<T>& roots)
{
	NodesT<T> out;
	out.reserve(roots.size());
	std::unordered_map<RepptrT<T>,NodeptrT<T>> unravelled;
	for (auto& root : roots)
	{
		auto tens = root->get_tensor();
		auto owners = ade::track_owners(tens);
		auto& rep = repr.reps_[tens.get()];
		unravel(unravelled, rep, owners);
		out.push_back(unravelled[rep]);
	}
	return out;
}

}

}

#endif // EAD_REPRESENT_HPP
