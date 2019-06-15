#include "ade/traveler.hpp"

#include "tag/group.hpp"
#include "tag/prop.hpp"

#include "experimental/opt/compares.hpp"

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
		for (auto group : other.groups_)
		{
			report_group(group.second, group.first);
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
	void report_any (ade::TensptrT tens, size_t any_id)
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

	void report_group (tag::SgraphptrT sg, size_t group_id)
	{
		auto it = groups_.find(group_id);
		if (groups_.end() == it || it->second == sg)
		{
			groups_.emplace(group_id, sg);
		}
		else
		{
			fail();
		}
	}

	void report_variadic (ade::TensptrT arg, size_t variadic_id)
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

	std::unordered_map<size_t,ade::TensptrT> bases_;

	std::unordered_map<size_t,tag::SgraphptrT> groups_;

	std::unordered_map<size_t,ade::TensT> variadics_;

private:
	bool success_ = true;
};

struct iWriter
{
	virtual ~iWriter (void) = default;

	virtual void write (Report& out,
		tag::SubgraphsT& subs, ade::TensptrT target) = 0;

	virtual size_t get_maxheight (void) const = 0;

	virtual std::string to_string (void) const = 0;
};

using WriterptrT = std::shared_ptr<iWriter>;

struct WriterArg final
{
	WriterArg (WriterptrT arg, ade::CoordptrT shaper, ade::CoordptrT coorder) :
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
			Report& temp_ctx = cand_pair.first;
			auto& field = cand_pair.second;
			for (size_t j = 0; j < nargs; ++j)
			{
				if (field[j])
				{
					continue;
				}
				if (false == ::opt::is_equal(args[j].get_shaper(), sub_rule.shaper_))
				{
					temp_ctx.fail();
					break;
				}
				if (false == ::opt::is_equal(args[j].get_coorder(), sub_rule.coorder_))
				{
					temp_ctx.fail();
					break;
				}
				sub_rule.arg_->write(temp_ctx, subs,
					args[j].get_tensor());
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
	const std::vector<ade::TensptrT>& args, const WriterArgsT& sub_rules)
{
	size_t nargs = args.size();
	CommCandsT candidates = {
		{Report{true}, std::vector<bool>(nargs, false)}
	};
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
struct ScalarWriter final : public iWriter
{
	ScalarWriter (double scalar) : scalar_(scalar) {}

	void write (Report& out,
		tag::SubgraphsT& subs, ade::TensptrT target) override
	{
		ScalarVisitor visitor(out, scalar_);
		target->accept(visitor);
	}

	size_t get_maxheight (void) const override
	{
		return 0;
	}

	std::string to_string (void) const override
	{
		return fmts::to_string(scalar_);
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
struct AnyWriter final : public iWriter
{
	AnyWriter (size_t id) : id_(id) {}

	void write (Report& out,
		tag::SubgraphsT& subs, ade::TensptrT target) override
	{
		out.report_any(target, id_);
	}

	size_t get_maxheight (void) const override
	{
		return 0;
	}

	std::string to_string (void) const override
	{
		return "id:" + fmts::to_string(id_);
	}

	size_t id_;
};

// e.g.: SUB(X, X)
struct FuncWriter final : public iWriter
{
	FuncWriter (std::string op, WriterArgsT sub_rules) :
		op_(op), sub_rules_(sub_rules) {}

	void write (Report& out,
		tag::SubgraphsT& subs, ade::TensptrT target) override
	{
		FuncVisitor visitor(out, subs, op_, sub_rules_);
		target->accept(visitor);
	}

	size_t get_maxheight (void) const override
	{
		std::vector<size_t> maxheights;
		maxheights.reserve(sub_rules_.size());
		std::transform(sub_rules_.begin(), sub_rules_.end(),
			std::back_inserter(maxheights),
			[](const WriterArg& arg)
			{
				return arg.arg_->get_maxheight();
			});
		auto it = std::max_element(maxheights.begin(), maxheights.end());
		if (maxheights.end() == it)
		{
			logs::fatal("function writer cannot have no sub rules");
		}
		return *it + 1;
	}

	std::string to_string (void) const override
	{
		std::vector<std::string> args;
		args.reserve(sub_rules_.size());
		std::transform(sub_rules_.begin(), sub_rules_.end(),
			std::back_inserter(args),
			[](const WriterArg& arg)
			{
				return arg.arg_->to_string();
			});
		return op_ + fmts::sprintf("(%s)", fmts::join(",",
			args.begin(), args.end()).c_str());
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
				if (false == ::opt::is_equal(args[i].get_shaper(), sub_rules_[i].shaper_))
				{
					report_->fail();
					return;
				}
				if (false == ::opt::is_equal(args[i].get_coorder(), sub_rules_[i].coorder_))
				{
					report_->fail();
					return;
				}
				sub_rules_[i].arg_->write(
					*report_, *subs_, args[i].get_tensor());
				if (false == report_->is_success())
				{
					return; // failure. so shortcircuit
				}
			}
		}

		Report* report_ = nullptr;

		tag::SubgraphsT* subs_ = nullptr;

		std::string op_;

		WriterArgsT sub_rules_;
	};
};

// e.g.: group:sum(SQUARE(SIN(X)),SQUARE(COS(Y)),..Z)
struct GroupWriter final : public iWriter
{
	GroupWriter (size_t group_id, std::string group,
		WriterArgsT sub_rules, size_t variadic_id) :
		group_id_(group_id), group_(group),
		sub_rules_(sub_rules), variadic_id_(variadic_id) {}

	void write (Report& out,
		tag::SubgraphsT& subs, ade::TensptrT target) override
	{
		GroupVisitor visitor(out, subs, group_id_,
			group_, sub_rules_, variadic_id_);
		target->accept(visitor);
	}

	size_t get_maxheight (void) const override
	{
		std::vector<size_t> maxheights;
		maxheights.reserve(sub_rules_.size());
		std::transform(sub_rules_.begin(), sub_rules_.end(),
			std::back_inserter(maxheights),
			[](const WriterArg& arg)
			{
				return arg.arg_->get_maxheight();
			});
		auto it = std::max_element(maxheights.begin(), maxheights.end());
		if (maxheights.end() == it)
		{
			return 1;
		}
		return *it + 1;
	}

	std::string to_string (void) const override
	{
		std::vector<std::string> args;
		args.reserve(sub_rules_.size());
		std::transform(sub_rules_.begin(), sub_rules_.end(),
			std::back_inserter(args),
			[](const WriterArg& arg)
			{
				return arg.arg_->to_string();
			});
		if (variadic_id_ != std::string::npos)
		{
			args.push_back(fmts::sprintf("..id:%d", variadic_id_));
		}
		return "group:" + group_ + fmts::sprintf("(%s)",
			fmts::join(",", args.begin(), args.end()).c_str());
	}

	size_t group_id_;

	std::string group_;

	WriterArgsT sub_rules_;

	size_t variadic_id_;

private:
	struct GroupVisitor final : public ade::iTraveler
	{
		GroupVisitor (Report& report, tag::SubgraphsT& subs, size_t group_id,
			std::string group, WriterArgsT sub_rules, size_t variadic_id) :
			report_(&report), subs_(&subs), group_id_(group_id),
			group_(group), sub_rules_(sub_rules), variadic_id_(variadic_id)
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
			std::vector<ade::TensptrT> args;
			args.reserve(nargs);
			for (auto cpair : children)
			{
				args.push_back(cpair.second);
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
			report_->report_group(sg, group_id_);
		}

		Report* report_ = nullptr;

		tag::SubgraphsT* subs_ = nullptr;

		size_t group_id_;

		std::string group_;

		WriterArgsT sub_rules_;

		size_t variadic_id_;
	};
};

}

}

#endif // OPT_RULE_WRITER_HPP
