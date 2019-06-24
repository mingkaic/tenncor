#include "ade/traveler.hpp"

#include "tag/group.hpp"

#include "opt/ivoter.hpp"

#ifndef OPT_MATCHER_HPP
#define OPT_MATCHER_HPP

namespace opt
{

const std::string group_prefix = "group:";

// approach: we have matcher, voters, and candidates
//  - matchers map:
//      - functors to voters (by functor's opname)
//      - leaves to ScalarCand (if it is immutable and scalar)
//      - anything that is not matched to a voter or candidate has no
//        candidate and marked as ANY
//  - voters are created by the parser, and only accept functor arguments
//      voters generate a list of candidates based on the arguments,
//      the list of candidates can be empty and marked as ANY
//  - candidates can be a scalar, an intermediate candidate,
//    or a convertible candidate. when built:
//      - scalar candidates take a constant scalar node
//      - convertible candidates take on it's converted subgraph
//      - intermediate candidates do nothing
//  Using the matcher, the optimizer makes a best attempt at
//  mapping tensor to zero to many candidates.
//  The optimizer is responsible for selecting the best candidates
struct Matcher final : public ade::iTraveler
{
	Matcher (void) = default;

	Matcher (const VoterPool& voters) : voters_(voters) {}

	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		if (false == util::has(candidates_, leaf))
		{
			if (tag::has_property(leaf, tag::immutable_tag))
			{
				std::string const_str = leaf->to_string();
				Symbol cand_key{CAND_TYPE::CONST, const_str};
				CandsT cands = {{cand_key, CtxsT{}}};
				if (is_scalar(leaf))
				{
					// match against scalar maps
					if (util::has(voters_.immutables_, const_str))
					{
						cands.emplace(cand_key, CtxsT{});
					}
				}
				candidates_.emplace(leaf, cands);
			}
			else
			{
				candidates_.emplace(leaf, CandsT{});
			}
		}
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (false == util::has(candidates_, func))
		{
			auto& children = func->get_children();
			for (auto& child : children)
			{
				child.get_tensor()->accept(*this);
			}

			if (std::all_of(children.begin(), children.end(),
				[this](const ade::FuncArg& child) -> bool
				{
					auto ctens = child.get_tensor().get();
					return util::has(this->candidates_[ctens],
						Symbol{CAND_TYPE::CONST, ctens->to_string()});
				}))
			{
				// all children are constants
				// therefore mark this as constant
				std::string const_str = func->to_string();
				candidates_.emplace(func, CandsT{
					{Symbol{CAND_TYPE::CONST, const_str}, CtxsT{}},
				});
				return;
			}

			CandsT out_cands;
			// functor
			std::string opname = func->get_opcode().name_;
			auto it = voters_.branches_.find(opname);
			if (voters_.branches_.end() != it)
			{
				CandArgsT args;
				args.reserve(children.size());
				for (auto& child : children)
				{
					auto ctens = child.get_tensor();
					args.push_back(CandArg{
						ctens,
						candidates_[ctens.get()],
						child.get_shaper(),
						child.get_coorder(),
					});
				}
				out_cands = it->second->inspect(args);
			}

			// do the same for functors that are the "head" of groups
			auto git = group_head_.find(func);
			if (group_head_.end() != git)
			{
				tag::SgraphptrT& sg = git->second;
				auto bit = voters_.branches_.find(group_prefix + sg->group_);
				if (voters_.branches_.end() != bit)
				{
					// todo: store sg->children_ as ade::ArgsT
					CandArgsT args;
					args.reserve(children.size());
					for (auto& sgcpair : sg->children_)
					{
						auto ctens = sgcpair.second;
						args.push_back(CandArg{
							ctens,
							candidates_[sgcpair.first],
							ade::identity,
							ade::CoordptrT(),
						});
					}
					CandsT group_cands = bit->second->inspect(args);
					out_cands.insert(group_cands.begin(), group_cands.end());
				}
			}

			candidates_.emplace(func, out_cands);
		}
	}

	// created by parser
	VoterPool voters_;

	// generated as visited
	std::unordered_map<ade::iTensor*,CandsT> candidates_;

	// heads for functors
	tag::SubgraphsT group_head_;
};

}

#endif // OPT_MATCHER_HPP
