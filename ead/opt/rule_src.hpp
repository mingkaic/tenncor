#include <regex>

#include "ead/generated/codes.hpp"

#include "ead/ead.hpp"

#include "ead/opt/represent.hpp"

#ifndef EAD_RULE_SRC_HPP
#define EAD_RULE_SRC_HPP

namespace ead
{

namespace opt
{

template <typename T>
struct RuleContext final
{
	bool emplace_varpair (iReprNode<T>* rep, size_t id)
	{
		bool found = rule_vars_.end() != rule_vars_.find(id);
		rule_vars_.emplace(id, rep);
		return found;
	}

	bool insert_variadic (const ReprArg<T>& arg, size_t variadic_id)
	{
		bool found = variadic_vars_.end() != variadic_vars_.find(variadic_id);
		variadic_vars_[variadic_id].push_back(arg);
		return found;
	}

	void emplace_edge (FuncRep<T>* parent, const ReprArg<T>& arg)
	{
		edges_.push_back(Edge{parent, arg});
	}

	// merge other's edges and varpairs into this
	void merge (const RuleContext<T>& other)
	{
		for (auto rule_var : other.rule_vars_)
		{
			rule_vars_.emplace(rule_var);
		}
		for (auto vari_var : other.variadic_vars_)
		{
			auto& vari_vars = variadic_vars_[vari_var.first];
			vari_vars.insert(vari_vars.end(),
				vari_var.second.begin(), vari_var.second.end());
		}
		edges_.insert(edges_.end(), other.edges_.begin(), other.edges_.end());
	}

	std::unordered_map<size_t,iReprNode<T>*> rule_vars_;

	std::unordered_map<size_t,RepArgsT<T>> variadic_vars_;

private:
	struct Edge
	{
		FuncRep<T>* parent_;

		ReprArg<T> arg_;
	};

	std::vector<Edge> edges_;
};

// e.g.: scalar_1.2
template <typename T>
struct ConstRule final : public iRuleNode<T>
{
	ConstRule (std::regex pattern) : pattern_(pattern) {}

	bool process (RuleContext<T>& ctx, ConstRep<T>* leaf) const override
	{
		return std::regex_match(leaf->get_identifier(), pattern_);
	}

	bool process (RuleContext<T>& ctx, LeafRep<T>* leaf) const override
	{
		return false;
	}

	bool process (RuleContext<T>& ctx, FuncRep<T>* func) const override
	{
		return false;
	}

	size_t get_minheight (void) const override
	{
		return 1;
	}

	std::regex pattern_;
};

// e.g.: X
template <typename T>
struct VarRule final : public iRuleNode<T>
{
	VarRule (size_t id) : id_(id) {}

	bool process (RuleContext<T>& ctx, ConstRep<T>* leaf) const override
	{
		return false;
	}

	bool process (RuleContext<T>& ctx, LeafRep<T>* leaf) const override
	{
		return ctx.emplace_varpair(leaf, id_);
	}

	bool process (RuleContext<T>& ctx, FuncRep<T>* func) const override
	{
		return ctx.emplace_varpair(func, id_);
	}

	size_t get_minheight (void) const override
	{
		return 1;
	}

	size_t id_;
};

// e.g.: SUB(X, X)
template <typename T>
struct FuncRule final : public iRuleNode<T>
{
	FuncRule (ade::Opcode op, RuleArgsT<T> sub_rules) :
		op_(op), sub_rules_(sub_rules) {}

	bool process (RuleContext<T>& ctx, ConstRep<T>* leaf) const override
	{
		return false;
	}

	bool process (RuleContext<T>& ctx, LeafRep<T>* leaf) const override
	{
		return false;
	}

	bool process (RuleContext<T>& ctx, FuncRep<T>* func) const override
	{
		if (func->op_.code_ != op_.code_)
		{
			return false;
		}

		auto& args = func->get_args();
		size_t nargs = args.size();
		if (sub_rules_.size() != nargs)
		{
			return false;
		}

		RuleContext<T> temp_ctx; // acts as a transaction
		for (size_t i = 0; i < nargs; ++i)
		{
			if (false == args[i].arg_->rulify(
				temp_ctx, sub_rules_[i].arg_))
			{
				return false;
			}
			temp_ctx.emplace_edge(func, args[i]);
		}
		ctx.merge(temp_ctx); // commit transaction
		return true;
	}

	size_t get_minheight (void) const override
	{
		std::vector<size_t> minheights;
		minheights.reserve(sub_rules_.size());
		std::transform(sub_rules_.begin(), sub_rules_.end(),
			std::back_inserter(minheights),
			[](const RuleArg<T>& sub_rule)
			{
				return sub_rule.arg_->get_minheight();
			});
		return *std::min_element(minheights.begin(), minheights.end()) + 1;
	}

	ade::Opcode op_;

	RuleArgsT<T> sub_rules_;
};

// e.g.: ADD(SQUARE(SIN(X)), SQUARE(COS(Y)), ..)
template <typename T>
struct VariadicFuncRule final : public iRuleNode<T>
{
	VariadicFuncRule (ade::Opcode op, RuleArgsT<T> sub_rules,
		size_t variadic_id) :
		op_(op), sub_rules_(sub_rules), variadic_id_(variadic_id)
	{
		assert(is_commutative(op.code_));
	}

	bool process (RuleContext<T>& ctx, ConstRep<T>* leaf) const override
	{
		return false;
	}

	bool process (RuleContext<T>& ctx, LeafRep<T>* leaf) const override
	{
		return false;
	}

	bool process (RuleContext<T>& ctx, FuncRep<T>* func) const override
	{
		if (func->op_.code_ != op_.code_)
		{
			return false;
		}

		auto& args = func->get_args();
		size_t nargs = args.size();
		size_t nrules = sub_rules_.size();
		if (nrules > nargs)
		{
			return false;
		}

		std::vector<std::pair<RuleContext<T>,std::vector<bool>>> candidates = {{
			RuleContext<T>(), std::vector<bool>(nargs, false)
		}};
		for (auto& sub_rule : sub_rules_)
		{
			std::vector<std::pair<RuleContext<T>,std::vector<bool>>> next_cands;
			for (auto& cand_pair : candidates)
			{
				auto& field = cand_pair.second;
				for (size_t j = 0; j < nargs; ++j)
				{
					if (field[j])
					{
						continue;
					}
					RuleContext<T> temp_ctx = cand_pair.first;
					if (args[j].arg_->rulify(temp_ctx, sub_rule.arg_))
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
				ctx.insert_variadic(args[j], variadic_id_);
			}
		}
		return true;
	}

	size_t get_minheight (void) const override
	{
		std::vector<size_t> minheights;
		minheights.reserve(sub_rules_.size() + 1);
		minheights.push_back(1); // 1 height for variadic argument
		std::transform(sub_rules_.begin(), sub_rules_.end(),
			std::back_inserter(minheights),
			[](const RuleArg<T>& sub_rule)
			{
				return sub_rule.arg_->get_minheight();
			});
		return *std::min_element(minheights.begin(), minheights.end()) + 1;
	}

	ade::Opcode op_;

	RuleArgsT<T> sub_rules_;

	size_t variadic_id_;
};

}

}

#endif // EAD_RULE_SRC_HPP
