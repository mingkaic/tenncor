#include "tag/group.hpp"
#include "tag/prop.hpp"

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
	{
		// both must be successful to merge meaningfully
		if (success_ != other.success_)
		{
			fail();
			return;
		}
		for (auto base : other.bases_)
		{
			report_any(base.second, base.first);
			if (false == success_)
			{
				return;
			}
		}
		for (auto ovari : other.variadics_)
		{
			auto& vari = variadics_[ovari.first];
			vari.insert(vari.end(),
				ovari.second.begin(), ovari.second.end());
		}
	}

	/// Succeed if any_id-tens pair does not conflict with existing pair,
	/// otherwise Fail
	void report_any (ade::iTensor* tens, size_t any_id)
	{
		auto it = bases_.find(any_id);
		if (bases_.end() == it || it->second == tens)
		{
			bases_.emplace(any_id, tens);
		}
		else
		{
			fail();
		}
	}

	void report_variadic (ade::iTensor* arg, size_t variadic_id)
	{
		variadics_[variadic_id].push_back(arg);
	}

	// absense of failure is success
	void fail (void)
	{
		success_ = false;
	}

	bool is_success (void) const
	{
		return success_;
	}

	std::unordered_map<size_t,ade::iTensor*> bases_;

	std::unordered_map<size_t,std::vector<ade::iTensor*>> variadics_;

private:
	bool success_ = true;
};

struct iWriter
{
	virtual ~iWriter (void) = default;

	virtual void write (Report& out,
		tag::SubgraphsT& subs, ade::iTensor* target) = 0;
};

using WriterptrT = std::shared_ptr<iWriter>;

struct WriterArg final
{
	WriterArg (WriterptrT arg,
		std::string shaper, std::string coorder) :
		arg_(arg), shaper_(shaper), coorder_(coorder)
	{
		if (nullptr == arg)
		{
			logs::fatal("created a rule argument with null argument");
		}
	}

	WriterptrT arg_;

	std::string shaper_;

	std::string coorder_;
};

using WriterArgsT = std::vector<WriterArg>;

using CommCandsT = std::vector<std::pair<Report,std::vector<bool>>>;

static bool coord_equal (ade::CoordptrT coord, std::string sform)
{
	if (ade::is_identity(coord.get()))
	{
		return sform.empty();
	}
	std::string cstr = coord->to_string();
	std::string simplified;
	simplified.reserve(cstr.size()); // optional, avoids buffer reallocations in the loop
	for(size_t i = 0; i < cstr.size(); ++i)
	{
		if (cstr[i] != fmts::arr_begin &&
			cstr[i] != fmts::arr_end)
		{
			simplified += cstr[i];
		}
		else if (cstr[i] == fmts::arr_delim)
		{
			simplified += ',';
		}
	}
	return simplified == sform;
}

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
			Report& temp_ctx = cand_pair.first;
			auto& field = cand_pair.second;
			for (size_t j = 0; j < nargs; ++j)
			{
				if (field[j])
				{
					continue;
				}
				if (false == coord_equal(args[j].get_shaper(), sub_rule.shaper_))
				{
					temp_ctx.fail();
					break;
				}
				if (false == coord_equal(args[j].get_coorder(), sub_rule.coorder_))
				{
					temp_ctx.fail();
					break;
				}
				sub_rule.arg_->write(temp_ctx, subs,
					args[j].get_tensor().get());
				if (temp_ctx.is_success())
				{
					std::vector<bool> temp_field = field;
					temp_field[j] = true;
					next_cands.push_back({temp_ctx, temp_field});
				}
			}
			if (false == temp_ctx.is_success())
			{
				break;
			}
		}
		candidates = next_cands;
	}
	return candidates;
}

static CommCandsT communtative_rule_match (tag::SubgraphsT& subs,
	const std::vector<ade::iTensor*>& args, const WriterArgsT& sub_rules)
{
	size_t nargs = args.size();
	CommCandsT candidates = {{Report{true}, std::vector<bool>(nargs, false)}};
	for (auto& sub_rule : sub_rules)
	{
		CommCandsT next_cands;
		for (auto& cand_pair : candidates)
		{
			Report& temp_ctx = cand_pair.first;
			auto& field = cand_pair.second;
			for (size_t j = 0; j < nargs; ++j)
			{
				if (field[j])
				{
					continue;
				}
				sub_rule.arg_->write(temp_ctx, subs, args[j]);
				if (temp_ctx.is_success())
				{
					std::vector<bool> temp_field = field;
					temp_field[j] = true;
					next_cands.push_back({temp_ctx, temp_field});
				}
			}
			if (false == temp_ctx.is_success())
			{
				break;
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

	void write (Report& out, tag::SubgraphsT& subs, ade::iTensor* target) override
	{
		ScalarVisitor visitor(out, scalar_);
		target->accept(visitor);
	}

	double scalar_;

private:
	struct ScalarVisitor final : public ade::iTraveler
	{
		ScalarVisitor (Report& report, double scalar) :
			report_(&report), scalar_(scalar)
		{
			if (nullptr == report_)
			{
				logs::fatal("attempting to process scalar without reporter");
			}
		}

		/// Implementation of iTraveler
		void visit (ade::iLeaf* leaf) override
		{
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
			report_->fail();
		}

		Report* report_ = nullptr;

		double scalar_;
	};
};

// e.g.: X
struct Any final : public iWriter
{
	Any (size_t id) : id_(id) {}

	void write (Report& out, tag::SubgraphsT& subs, ade::iTensor* target) override
	{
		AnyVisitor visitor(out, id_);
		target->accept(visitor);
	}

	size_t id_;

private:
	struct AnyVisitor final : public ade::iTraveler
	{
		AnyVisitor (Report& report, size_t id) :
			report_(&report), id_(id)
		{
			if (nullptr == report_)
			{
				logs::fatal("attempting to process any without reporter");
			}
		}

		/// Implementation of iTraveler
		void visit (ade::iLeaf* leaf) override
		{
			report_->report_any(leaf, id_);
		}

		/// Implementation of iTraveler
		void visit (ade::iFunctor* func) override
		{
			report_->report_any(func, id_);
		}

		Report* report_ = nullptr;

		size_t id_;
	};
};

// e.g.: SUB(X, X)
struct Func final : public iWriter
{
	Func (std::string op, WriterArgsT sub_rules) :
		op_(op), sub_rules_(sub_rules) {}

	void write (Report& out, tag::SubgraphsT& subs, ade::iTensor* target) override
	{
		FuncVisitor visitor(out, subs, op_, sub_rules_);
		target->accept(visitor);
	}

	std::string op_;

	WriterArgsT sub_rules_;

private:
	struct FuncVisitor final : public ade::iTraveler
	{
		FuncVisitor (Report& report, tag::SubgraphsT& subs,
			std::string op, WriterArgsT sub_rules) :
			report_(&report), subs_(&subs),
			op_(op), sub_rules_(sub_rules)
		{
			if (nullptr == report_)
			{
				logs::fatal("attempting to process functor without reporter");
			}
			if (nullptr == subs_)
			{
				logs::fatal("attempting to process functor without subgraphs");
			}
		}

		/// Implementation of iTraveler
		void visit (ade::iLeaf* leaf) override
		{
			report_->fail();
		}

		/// Implementation of iTraveler
		void visit (ade::iFunctor* func) override
		{
			if (func->get_opcode().name_ != op_)
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
				if (false == coord_equal(args[i].get_shaper(), sub_rules_[i].shaper_))
				{
					report_->fail();
					return;
				}
				if (false == coord_equal(args[i].get_coorder(), sub_rules_[i].coorder_))
				{
					report_->fail();
					return;
				}
				sub_rules_[i].arg_->write(
					*report_, *subs_, args[i].get_tensor().get());
				if (false == report_->is_success())
				{
					return; // failure. so shortcircuit
				}
			}
		}

		tag::SubgraphsT* subs_ = nullptr;

		Report* report_ = nullptr;

		std::string op_;

		WriterArgsT sub_rules_;
	};
};

// e.g.: group:sum(SQUARE(SIN(X)),SQUARE(COS(Y)),..Z)
struct Group final : public iWriter
{
	Group (std::string group, WriterArgsT sub_rules, size_t variadic_id) :
		group_(group), sub_rules_(sub_rules),
		variadic_id_(variadic_id) {}

	void write (Report& out, tag::SubgraphsT& subs, ade::iTensor* target) override
	{
		GroupVisitor visitor(out, subs,
			group_, sub_rules_, variadic_id_);
		target->accept(visitor);
	}

	std::string group_;

	WriterArgsT sub_rules_;

	size_t variadic_id_;

private:
	struct GroupVisitor final : public ade::iTraveler
	{
		GroupVisitor (Report& report, tag::SubgraphsT& subs,
			std::string group, WriterArgsT sub_rules,
			size_t variadic_id) :
			report_(&report), subs_(&subs),
			group_(group), sub_rules_(sub_rules),
			variadic_id_(variadic_id)
		{
			if (nullptr == report_)
			{
				logs::fatal("attempting to process variadic functor without reporter");
			}
			if (nullptr == subs_)
			{
				logs::fatal("attempting to process variadic functor without subgraphs");
			}
		}

		/// Implementation of iTraveler
		void visit (ade::iLeaf* leaf) override
		{
			report_->fail();
		}

		/// Implementation of iTraveler
		void visit (ade::iFunctor* func) override
		{
			auto it = subs_->find(func);
			if (subs_->end() == it || it->second->group_ != group_)
			{
				report_->fail();
				return;
			}
			tag::SgraphptrT& sg = it->second;

			std::unordered_map<ade::iTensor*,ade::TensptrT>&
			children = sg->children_;
			size_t nargs = children.size();
			std::vector<ade::iTensor*> args;
			args.reserve(nargs);
			for (auto cpair : children)
			{
				args.push_back(cpair.first);
			}
			if (sub_rules_.size() > nargs)
			{
				report_->fail();
				return;
			}
			// nonvariadic arguments have to match all args
			if (variadic_id_ == std::string::npos &&
				sub_rules_.size() != nargs)
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
				logs::debugf("%d candidates found for group",
					candidates.size());
			}
			auto& report = candidates[0].first;
			auto& field = candidates[0].second;
			for (size_t j = 0; j < nargs; ++j)
			{
				if (false == field[j])
				{
					// only occurs for variadic groups
					report.report_variadic(args[j], variadic_id_);
				}
			}
			report_->merge(report); // commit transaction
		}

		tag::SubgraphsT* subs_ = nullptr;

		Report* report_ = nullptr;

		std::string group_;

		WriterArgsT sub_rules_;

		size_t variadic_id_;
	};
};

}

}

#endif // OPT_RULE_WRITER_HPP
