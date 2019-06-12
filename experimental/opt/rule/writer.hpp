#include "ade/traveler.hpp"

#ifndef OPT_RULE_WRITER_HPP
#define OPT_RULE_WRITER_HPP

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

	// absense of failure is success
	void fail (void)
	{
		success_ = false;
	}

	bool is_success (void) const
	{
		return success_;
	}

private:
	bool success_ = true;
};

struct iWriter : public ade::iTraveler
{
	virtual ~iWriter (void) = default;

	virtual void write (Report& out,
		tag::SubgraphsT& subs, ade::iTensor* target) = 0;
};

using WriterptrT = std::shared_ptr<iWriter>;

struct WriterArg final
{
	WriterArg (WriterptrT arg,
		ade::CoordptrT shaper, ade::CoordptrT coorder) :
		arg_(arg), shaper_(shaper), coorder_(coorder)
	{
		if (nullptr == arg)
		{
			logs::fatal("created a rule argument with null argument");
		}
	}

	WriterptrT arg_;

	ade::CoordptrT shaper_;

	ade::CoordptrT coorder_;
};

using WriterArgsT = std::vector<WriterArg>;

using CommCandsT = std::vector<std::pair<Report,std::vector<bool>>>;

// todo: make this much more effcient (currently brute-force)
// consider sort then match (need to make rule nodes hashed similar to actual tensors)
static CommCandsT communtative_rule_match (tag::SubgraphsT& subs,
	const ade::ArgsT& args, const WriterArgsT& sub_rules)
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
				sub_rule.arg_->write(temp_ctx, subs,
					args[j].get_tensor().get());
				if (temp_ctx.is_success())
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
struct Scalar final : public iWriter
{
	Scalar (double scalar) : scalar_(scalar) {}

	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		if (nullptr == report_)
		{
			logs::fatal("attempting to match scalar leaf without reporter");
		}

		if (tag::has_property(leaf, tag::immutable_tag))
		{
			ade::Shape shape = leaf->shape();
			char* data = (char*) leaf->data();
			size_t n = shape.n_elems();
			size_t perbytes = leaf->nbytes() / n;
			for (size_t i = 1; i < n; ++i)
			{
				if (false == std::equal(data, data + perbytes,
					data + i * perbytes))
				{
					// not scalar
					report_->fail();
					return;
				}
			}

			if (fmts::to_string(scalar_) != leaf->to_string())
			{
				report_->fail();
			}
		}
		else
		{
			report_->fail();
		}
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (nullptr == report_)
		{
			logs::fatal("attempting to match scalar functor without reporter");
		}
		report_->fail();
	}

	void write (Report& out, tag::SubgraphsT& subs, ade::iTensor* target)
	{
		report_ = &out;
		target->accept(*this);
		report_ = nullptr;
	}

	double scalar_;

	Report* report_ = nullptr;
};

// e.g.: X
struct Any final : public iWriter
{
	Any (size_t id) : id_(id) {}

	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		if (nullptr == report_)
		{
			logs::fatal("attempting to match any leaf without reporter");
		}
		report_->report_any(leaf, id_);
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (nullptr == report_)
		{
			logs::fatal("attempting to match any functor without reporter");
		}
		report_->report_any(func, id_);
	}

	void write (Report& out, tag::SubgraphsT& subs, ade::iTensor* target)
	{
		report_ = &out;
		target->accept(*this);
		report_ = nullptr;
	}

	size_t id_;

	Report* report_ = nullptr;
};

// e.g.: SUB(X, X)
struct Func final : public iWriter
{
	Func (std::string op, WriterArgsT sub_rules) :
		// todo: parse group
		op_(ade::Opcode{op, 0}), sub_rules_(sub_rules) {}

	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		if (nullptr == report_)
		{
			logs::fatal("attempting to match func leaf without reporter");
		}
		report_->fail();
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (nullptr == report_)
		{
			logs::fatal("attempting to match func functor without reporter");
		}
		if (nullptr == subs_)
		{
			logs::fatal("attempting to match functor without subgraphs");
		}
		if (func->get_opcode().code_ != op_.code_)
		{
			report_->fail();
			return;
		}

		auto& args = func->get_children();
		size_t nargs = args.size();
		if (sub_rules_.size() != nargs)
		{
			report_->fail();
			return;
		}

		if (tag::has_property(func, tag::commutative_tag))
		{
			CommCandsT candidates = communtative_rule_match(*subs_, args, sub_rules_);
			if (candidates.empty()) // we've found no candidates
			{
				report_->fail();
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
			sub_rules_[i].arg_->write(
				*report_, *subs_, args[i].get_tensor().get());
			if (false == report_->is_success())
			{
				return; // failure. so shortcircuit
			}
		}
	}

	void write (Report& out, tag::SubgraphsT& subs, ade::iTensor* target)
	{
		subs_ = &subs;
		report_ = &out;
		target->accept(*this);
		subs_ = nullptr;
		report_ = nullptr;
	}

	ade::Opcode op_;

	WriterArgsT sub_rules_;

	tag::SubgraphsT* subs_ = nullptr;

	Report* report_ = nullptr;
};

// e.g.: ADD(SQUARE(SIN(X)),SQUARE(COS(Y)),..Z)
struct VariadicFunc final : public iWriter
{
	VariadicFunc (std::string op,
		WriterArgsT sub_rules, size_t variadic_id) :
		// todo: parse group
		op_(ade::Opcode{op, 0}), sub_rules_(sub_rules), variadic_id_(variadic_id) {}

	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		if (nullptr == report_)
		{
			logs::fatal("attempting to match variadic leaf without reporter");
		}
		report_->fail();
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (nullptr == report_)
		{
			logs::fatal("attempting to match variadic functor without reporter");
		}
		if (nullptr == subs_)
		{
			logs::fatal("attempting to match functor without subgraphs");
		}
		if (func->get_opcode().code_ != op_.code_)
		{
			report_->fail();
			return;
		}

		auto& args = func->get_children();
		size_t nargs = args.size();
		if (sub_rules_.size() > nargs)
		{
			report_->fail();
			return;
		}

		CommCandsT candidates = communtative_rule_match(*subs_, args, sub_rules_);
		if (candidates.empty()) // we've found no candidates
		{
			report_->fail();
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

	void write (Report& out, tag::SubgraphsT& subs, ade::iTensor* target)
	{
		subs_ = &subs;
		report_ = &out;
		target->accept(*this);
		subs_ = nullptr;
		report_ = nullptr;
	}

	ade::Opcode op_;

	WriterArgsT sub_rules_;

	size_t variadic_id_;

	tag::SubgraphsT* subs_ = nullptr;

	Report* report_ = nullptr;
};

}

}

#endif // OPT_RULE_WRITER_HPP
