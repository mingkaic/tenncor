#include "ead/generated/codes.hpp"

#include "ead/ead.hpp"

#include "ead/opt/irep.hpp"

#ifndef EAD_RULE_HPP
#define EAD_RULE_HPP

namespace ead
{

namespace opt
{

struct RuleContext
{
	bool emplace_varpair (iReprNode* rep, size_t id)
	{
		bool found = rule_vars_.end() != rule_vars_.find(id);
		rule_vars_.emplace(id, rep);
		return found;
	}

	void emplace_edge (FuncRep* parent, const ReprArg& arg)
	{
		edges_.push_back(Edge{parent, arg});
	}

	// merge other's edges and varpairs into this
	void merge (const RuleContext& other)
	{
		for (auto rule_var : other.rule_vars_)
		{
			rule_vars_.emplace(rule_var);
		}
		edges_.insert(edges_.end(), other.edges_.begin(), other.edges_.end());
	}

	std::unordered_map<size_t,iReprNode*> rule_vars_;

private:
	struct Edge
	{
		FuncRep* parent_;

		ReprArg arg_;
	};

	std::vector<Edge> edges_;
};

// e.g.: X
struct VarRule final : public iRuleNode
{
	VarRule (size_t id) : id_(id) {}

	bool process (RuleContext& ctx, LeafRep* leaf) const override
	{
		return ctx.emplace_varpair(leaf, id_);
	}

	bool process (RuleContext& ctx, FuncRep* func) const override
	{
		return ctx.emplace_varpair(func, id_);
	}

	size_t id_;
};

// e.g.: SUB(X, X)
struct FuncRule final : public iRuleNode
{
	FuncRule (ade::Opcode op, RuleArgsT sub_rules) :
		op_(op), sub_rules_(sub_rules) {}

	bool process (RuleContext& ctx, LeafRep* leaf) const override
	{
		return false;
	}

	bool process (RuleContext& ctx, FuncRep* func) const override
	{
		if (func->op_.code_ != op_.code_)
		{
			return false;
		}

		auto& args = func->args_;
		size_t nargs = args.size();
		if (sub_rules_.size() != nargs)
		{
			return false;
		}

		RuleContext temp_ctx; // acts as a transaction
		for (size_t i = 0; i < nargs; ++i)
		{
			if (false == args[i].arg_->rulify(
				temp_ctx, *sub_rules_[i]))
			{
				return false;
			}
			temp_ctx.emplace_edge(func, args[i]);
		}
		ctx.merge(temp_ctx); // commit transaction
		return true;
	}

	ade::Opcode op_;

	RuleArgsT sub_rules_;
};

// e.g.: ADD(SQUARE(SIN(X)), SQUARE(COS(Y)), ..)
struct VariadicFuncRule final : public iRuleNode
{
	VariadicFuncRule (ade::Opcode op, RuleArgsT sub_rules) :
		op_(op), sub_rules_(sub_rules) {}

	bool process (RuleContext& ctx, LeafRep* leaf) const override
	{
		return false;
	}

	bool process (RuleContext& ctx, FuncRep* func) const override
	{
		if (func->op_.code_ != op_.code_)
		{
			return false;
		}

		auto& args = func->args_;
		size_t nargs = args.size();
		size_t nrules = sub_rules_.size();
		if (nrules > nargs)
		{
			return false;
		}

		RuleContext temp_ctx; // acts as a transaction
		for (size_t i = 0, j = 0; i < nargs && j < nrules; ++i)
		{
			auto& arg = args[i];
			if (args[i].arg_->rulify(
				temp_ctx, *sub_rules_[j]))
			{
				temp_ctx.emplace_edge(func, args[i].arg_.get());
				++j;
			}
		}
		if (j < nrules) // we exhausted all args without finding match for subs
		{
			return false;
		}
		ctx.merge(temp_ctx); // commit transaction
		return true;
	}

	ade::Opcode op_;

	RuleArgsT sub_rules_;
};

}

}

#endif // EAD_RULE_HPP
