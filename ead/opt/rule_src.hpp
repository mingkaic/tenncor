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
	/// Return true if id-rep pair does not conflict with existing pair,
	/// otherwise return false
	bool emplace_varpair (iReprNode<T>* rep, size_t id)
	{
		auto it = rule_vars_.find(id);
		bool noconflict = rule_vars_.end() == it || it->second == rep;
		rule_vars_.emplace(id, rep);
		return noconflict;
	}

	/// Record arg under variadic_id bucket of all variadic_vars
	void insert_variadic (const ReprArg<T>& arg, size_t variadic_id)
	{
		variadic_vars_[variadic_id].push_back(arg);
	}

	/// Record parent-argument edge
	void emplace_edge (FuncRep<T>* parent, const ReprArg<T>& arg)
	{
		edges_.push_back(ContextEdge{parent, arg});
	}

	/// Return true if successfully merge variables
	/// and edges without conflicts, otherwise return false
	bool merge (const RuleContext<T>& other)
	{
		bool noconflicts = true;
		for (auto rule_var : other.rule_vars_)
		{
			auto it = rule_vars_.find(rule_var.first);
			noconflicts = noconflicts && (rule_vars_.end() == it ||
				it->second == rule_var.second);
			rule_vars_.emplace(rule_var);
		}
		for (auto vari_var : other.variadic_vars_)
		{
			auto& vari_vars = variadic_vars_[vari_var.first];
			vari_vars.insert(vari_vars.end(),
				vari_var.second.begin(), vari_var.second.end());
		}
		edges_.insert(edges_.end(), other.edges_.begin(), other.edges_.end());
		return noconflicts;
	}

	std::unordered_map<size_t,iReprNode<T>*> rule_vars_;

	std::unordered_map<size_t,RepArgsT<T>> variadic_vars_;

private:
	struct ContextEdge
	{
		FuncRep<T>* parent_;

		ReprArg<T> arg_;
	};

	std::vector<ContextEdge> edges_;
};

// e.g.: scalar_1.2
template <typename T>
struct ConstRule final : public iRuleNode<T>
{
	ConstRule (std::string pattern) : pattern_(pattern) {}

	bool process (RuleContext<T>& ctx, ConstRep<T>* leaf) const override
	{
		return std::regex_match(leaf->get_identifier(), std::regex(pattern_));
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

	std::string pattern_;
};

// e.g.: X
template <typename T>
struct AnyRule final : public iRuleNode<T>
{
	AnyRule (size_t id) : id_(id) {}

	bool process (RuleContext<T>& ctx, ConstRep<T>* leaf) const override
	{
		return ctx.emplace_varpair(leaf, id_);
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

template <typename T>
using CommCandsT = std::vector<std::pair<RuleContext<T>,std::vector<bool>>>;

// todo: make this much more effcient (currently brute-force)
template <typename T>
static CommCandsT<T> communtative_rule_match (
	const RepArgsT<T>& args, const RuleArgsT<T>& sub_rules)
{
	size_t nargs = args.size();
	CommCandsT<T> candidates = {{
		RuleContext<T>(), std::vector<bool>(nargs, false)
	}};
	for (auto& sub_rule : sub_rules)
	{
		CommCandsT<T> next_cands;
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
					std::vector<bool> temp_field = field;
					temp_field[j] = true;
					next_cands.push_back({temp_ctx, temp_field});
				}
			}
		}
		candidates = next_cands;
	}
	return candidates;
}

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

		if (is_commutative(func->op_.code_))
		{
			CommCandsT<T> candidates = communtative_rule_match(args, sub_rules_);
			if (candidates.empty()) // we've found no candidates
			{
				return false;
			}
			if (candidates.size() > 1)
			{
				// multiple candidates
				logs::debugf("%d candidates found for func rule",
					candidates.size());
			}
			return ctx.merge(candidates[0].first); // commit transaction
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
		return ctx.merge(temp_ctx); // commit transaction
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

// e.g.: ADD(SQUARE(SIN(X)),SQUARE(COS(Y)),..Z)
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

		CommCandsT<T> candidates = communtative_rule_match(args, sub_rules_);
		if (candidates.empty()) // we've found no candidates
		{
			return false;
		}
		if (candidates.size() > 1)
		{
			// multiple candidates
			logs::debugf("%d candidates found for variadic rule",
				candidates.size());
		}
		auto& field = candidates[0].second;
		for (size_t j = 0; j < nargs; ++j)
		{
			if (false == field[j])
			{
				candidates[0].first.insert_variadic(args[j], variadic_id_);
			}
		}
		return ctx.merge(candidates[0].first); // commit transaction
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
