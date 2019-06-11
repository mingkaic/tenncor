#include "ade/traveler.hpp"

#ifndef OPT_RULE_HPP
#define OPT_RULE_HPP

namespace opt
{

namespace rule
{

struct Report
{
	Report (void) = default;

	Report (bool success) : success_(success) {}

	void merge (const Report& other)
	{}

	void report_any (ade::iTensor* tens, size_t any_id)
	{}

	void report_variadic (const ade::FuncArg& arg, size_t variadic_id)
	{}

	bool success_ = false;
};

struct iNode : public ade::iTraveler
{
	virtual ~iNode (void) = default;

	virtual void get_report (Report& out, ade::iTensor* target) = 0;
};

using NodeptrT = std::shared_ptr<iNode>;

struct Arg final
{
	Arg (NodeptrT arg,
		ade::CoordptrT shaper, ade::CoordptrT coorder) :
		arg_(arg), shaper_(shaper), coorder_(coorder)
	{
		if (nullptr == arg)
		{
			logs::fatal("created a rule argument with null argument");
		}
	}

	NodeptrT arg_;

	ade::CoordptrT shaper_;

	ade::CoordptrT coorder_;
};

using ArgsT = std::vector<Arg>;

using CommCandsT = std::vector<std::pair<Report,std::vector<bool>>>;

// todo: make this much more effcient (currently brute-force)
// consider sort then match (need to make rule nodes hashed similar to actual tensors)
static CommCandsT communtative_rule_match (
	const ade::ArgsT& args, const ArgsT& sub_rules)
{
	size_t nargs = args.size();
	CommCandsT candidates = {{Report{true}, std::vector<bool>(nargs, false)}};
	for (auto& sub_rule : sub_rules)
	{
		CommCandsT next_cands;
		for (auto& cand_pair : candidates)
		{
			auto& field = cand_pair.second;
			for (size_t j = 0; j < nargs; ++j)
			{
				if (field[j])
				{
					continue;
				}
				Report& temp_ctx = cand_pair.first;
				sub_rule.arg_->get_report(temp_ctx,
					args[j].get_tensor().get());
				if (temp_ctx.success_)
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

// e.g.: 1.2
struct Scalar final : public iNode
{
	Scalar (std::string pattern) : pattern_(pattern) {}

	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		if (tag::has_property(leaf, tag::immutable_tag))
		{
			ade::Shape shape = leaf->shape();
			age::_GENERATED_DTYPE dtype =
				(age::_GENERATED_DTYPE) leaf->type_code();
			std::vector<float> data;
			age::type_convert(data, leaf->data(), dtype, shape.n_elems());

			std::string data_str;
			size_t n = shape.n_elems();
			if (std::all_of(data.begin() + 1, data.end(),
				[&](const double& e) { return e == data[0]; }))
			{
				// is a scalar
				double scalar = data[0];
				if (scalar == 0) // avoid -0
				{
					data_str = "_scalar_0";
				}
				else
				{
					data_str = "_scalar_" + fmts::to_string(scalar);
				}
			}
			else
			{
				// todo: make encoding better so we don't lose precision
				data_str = "_array_" + fmts::join("",
					data.begin(), data.end());
			}

			report_->success_ = std::regex_match(
				fmts::join("", shape.begin(), shape.end()) + data_str,
				std::regex(pattern_));
		}
		else
		{
			report_->success_ = false;
		}
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		report_->success_ = false;
	}

	void get_report (Report& out, ade::iTensor* target)
	{
		report_ = &out;
		target->accept(*this);
		report_ = nullptr;
	}

	std::string pattern_;

	Report* report_ = nullptr;
};

// e.g.: X
struct Any final : public iNode
{
	Any (size_t id) : id_(id) {}

	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		report_->report_any(leaf, id_);
		report_->success_ = true;
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		report_->report_any(func, id_);
		report_->success_ = true;
	}

	void get_report (Report& out, ade::iTensor* target)
	{
		report_ = &out;
		target->accept(*this);
		report_ = nullptr;
	}

	size_t id_;

	Report* report_ = nullptr;
};

// e.g.: SUB(X, X)
struct Func final : public iNode
{
	Func (ade::Opcode op, ArgsT sub_rules) :
		op_(op), sub_rules_(sub_rules) {}

	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		report_->success_ = false;
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (func->get_opcode().code_ != op_.code_)
		{
			report_->success_ = false;
			return;
		}

		auto& args = func->get_children();
		size_t nargs = args.size();
		if (sub_rules_.size() != nargs)
		{
			report_->success_ = false;
			return;
		}

		if (tag::has_property(func, tag::commutative_tag))
		{
			CommCandsT candidates = communtative_rule_match(args, sub_rules_);
			if (candidates.empty()) // we've found no candidates
			{
				report_->success_ = false;
				return;
			}
			if (candidates.size() > 1)
			{
				// multiple candidates
				logs::debugf("%d candidates found for func rule",
					candidates.size());
			}
			report_->merge(candidates[0].first); // commit transaction
			return;
		}
		for (size_t i = 0; i < nargs; ++i)
		{
			sub_rules_[i].arg_->get_report(
				*report_, args[i].get_tensor().get());
			if (false == report_->success_)
			{
				return; // failure
			}
		}
		report_->success_ = true;
	}

	void get_report (Report& out, ade::iTensor* target)
	{
		report_ = &out;
		target->accept(*this);
		report_ = nullptr;
	}

	ade::Opcode op_;

	ArgsT sub_rules_;

	Report* report_ = nullptr;
};

// e.g.: ADD(SQUARE(SIN(X)),SQUARE(COS(Y)),..Z)
struct VariadicFunc final : public iNode
{
	VariadicFunc (ade::Opcode op,
		ArgsT sub_rules, size_t variadic_id) :
		op_(op), sub_rules_(sub_rules), variadic_id_(variadic_id) {}

	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		report_->success_ = false;
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (func->get_opcode().code_ != op_.code_)
		{
			report_->success_ = false;
			return;
		}

		auto& args = func->get_children();
		size_t nargs = args.size();
		if (sub_rules_.size() > nargs)
		{
			report_->success_ = false;
			return;
		}

		CommCandsT candidates = communtative_rule_match(args, sub_rules_);
		if (candidates.empty()) // we've found no candidates
		{
			report_->success_ = false;
			return;
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
				candidates[0].first.report_variadic(args[j], variadic_id_);
			}
		}
		report_->merge(candidates[0].first); // commit transaction
	}

	void get_report (Report& out, ade::iTensor* target)
	{
		report_ = &out;
		target->accept(*this);
		report_ = nullptr;
	}

	ade::Opcode op_;

	ArgsT sub_rules_;

	size_t variadic_id_;

	Report* report_ = nullptr;
};

}

}

#endif // OPT_RULE_HPP
