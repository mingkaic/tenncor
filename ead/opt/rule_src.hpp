#include "ead/generated/codes.hpp"

#include "ead/ead.hpp"

#include "ead/opt/irep.hpp"

#ifndef EAD_RULE_SRC_HPP
#define EAD_RULE_SRC_HPP

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

	bool emplace_variadic_pair (const ReprArg& arg, size_t variadic_id)
	{
		bool found = variadic_vars_.end() != variadic_vars_.find(variadic_id);
		variadic_vars_.emplace(variadic_id, arg);
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

	std::unordered_map<size_t,ReprArg> variadic_vars_;

private:
	struct Edge
	{
		FuncRep* parent_;

		ReprArg arg_;
	};

	std::vector<Edge> edges_;
};

// e.g.: scalar_1.2
struct ConstRule final : public iRuleNode
{
	ConstRule (size_t id, std::regex pattern) : id_(id), pattern_(pattern) {}

	bool process (RuleContext& ctx, ConstRep* leaf) const override
	{
		if (std::regex_match(leaf->get_identifier(), pattern_))
		{
			return return ctx.emplace_varpair(leaf, id_);
		}
		return false;
	}

	bool process (RuleContext& ctx, LeafRep* leaf) const override
	{
		return false;
	}

	bool process (RuleContext& ctx, FuncRep* func) const override
	{
		return false;
	}

	size_t get_height (void) const override
	{
		return 1;
	}

	size_t id_;

	std::regex pattern_;
};

// e.g.: X
struct VarRule final : public iRuleNode
{
	VarRule (size_t id) : id_(id) {}

	bool process (RuleContext& ctx, ConstRep* leaf) const override
	{
		return false;
	}

	bool process (RuleContext& ctx, LeafRep* leaf) const override
	{
		return ctx.emplace_varpair(leaf, id_);
	}

	bool process (RuleContext& ctx, FuncRep* func) const override
	{
		return ctx.emplace_varpair(func, id_);
	}

	size_t get_height (void) const override
	{
		return 1;
	}

	size_t id_;
};

// e.g.: SUB(X, X)
struct FuncRule final : public iRuleNode
{
	FuncRule (ade::Opcode op, RuleArgsT sub_rules) :
		op_(op), sub_rules_(sub_rules) {}

	bool process (RuleContext& ctx, ConstRep* leaf) const override
	{
		return false;
	}

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

	size_t get_height (void) const override
	{
		size_t max_height = 0;
		for (auto& sub_rule : sub_rules_)
		{
			size_t height = sub_rule.arg_->get_height();
			if (max_height < height)
			{
				max_height = height;
			}
		}
		return max_height + 1;
	}

	ade::Opcode op_;

	RuleArgsT sub_rules_;
};

// e.g.: ADD(SQUARE(SIN(X)), SQUARE(COS(Y)), ..)
struct VariadicFuncRule final : public iRuleNode
{
	VariadicFuncRule (ade::Opcode op, RuleArgsT sub_rules) :
		op_(op), sub_rules_(sub_rules)
	{
		assert(is_commutative(op.code_));
	}

	bool process (RuleContext& ctx, ConstRep* leaf) const override
	{
		return false;
	}

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

		std::vector<std::pair<RuleContext,std::vector<bool>>> candidates = {{
			RuleContext(), std::vector<size_t>(false, nargs)
		}};
		for (size_t j = 0; j < nargs; ++j)
		{
			RuleContext temp_ctx; // acts as a transaction
			if (args[j].arg_->rulify(temp_ctx, *sub_rules_[0]))
			{
				std::vector<bool> field(false, nargs);
				field[j] = true;
				candidates.push_back({temp_ctx, field});
			}
		}
		for (size_t i = 0; i < nrules; ++i)
		{
			auto& sub_rule = sub_rules_[i];
			std::vector<std::pair<RuleContext,std::vector<bool>>> next_cands;
			for (auto& cand_pair : candidates)
			{
				auto& field = cand_pair.second;
				for (size_t j = 0; j < nargs; ++j)
				{
					if (field[j])
					{
						continue;
					}
					RuleContext temp_ctx = cand_pair.first;
					if (args[j].arg_->rulify(temp_ctx, *sub_rule))
					{
						auto temp_field = field;
						temp_field[j] = true;
						next_cands.push_back({temp_ctx, temp_field});
					}
				}
			}
			candidates = next_cands;
		}
		if (candidates.empty()) // we've found no candidates
		{
			return false;
		}
		if (candidates.size() > 1)
		{
			// log multiple candidates
		}
		ctx.merge(candidates[0].first); // commit transaction
		auto& field = candidates[0].second;
		for (size_t j = 0; j < nargs; ++j)
		{
			if (false == field[j])
			{
				ctx.emplace_variadic_pair(args[j], variadic_id_);
			}
		}
		return true;
	}

	size_t get_height (void) const override
	{
		size_t max_height = 0;
		for (auto& sub_rule : sub_rules_)
		{
			size_t height = sub_rule.arg_->get_height();
			if (max_height < height)
			{
				max_height = height;
			}
		}
		return max_height + 1;
	}

	ade::Opcode op_;

	RuleArgsT sub_rules_;

	size_t variadic_id_;
};

}

}

#endif // EAD_RULE_SRC_HPP
