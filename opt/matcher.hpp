///
/// matcher.hpp
/// opt
///
/// Purpose:
/// Implement matcher traveler for gathering
/// conversion candidates from a TEQ graph
///

#include "teq/traveler.hpp"

#include "tag/group.hpp"

#include "opt/ivoter.hpp"

#ifndef OPT_MATCHER_HPP
#define OPT_MATCHER_HPP

namespace opt
{

/// String to prefix group types
const std::string group_prefix = "group:";

///	Approach: we have matcher, voters, and candidates
///	- matchers map:
///		- functors to voters (by functor's opname)
///		- leaves to ScalarCand (if it is immutable and scalar)
///		- anything that is not matched to a voter or candidate has no
///		  candidate and marked as ANY
///	- voters are created by the parser, and only accept functor arguments
///		voters generate a list of candidates based on the arguments,
///		the list of candidates can be empty and marked as ANY
///	- candidates can be a scalar, an intermediate candidate,
///	  or a convertible candidate. when built:
///		- scalar candidates take a constant scalar node
///		- convertible candidates take on it's converted subgraph
///		- intermediate candidates do nothing
///	Using the matcher, the optimizer makes a best attempt at
///	mapping tensor to zero to many candidates.
///	The optimizer is responsible for selecting the best candidates
struct Matcher final : public teq::iTraveler
{
	Matcher (void) = default;

	Matcher (const VoterPool& voters) : voters_(voters) {}

	/// Implementation of iTraveler
	void visit (teq::iLeaf* leaf) override
	{
		if (false == estd::has(candidates_, leaf))
		{
			if (tag::get_property_reg().has_property(leaf, tag::immutable_tag))
			{
				std::string const_str = leaf->to_string();
				CandsT cands = {
					{Symbol{CAND_TYPE::CONST, const_str}, CtxsT{}}
				};
				if (is_scalar(leaf))
				{
					// match against scalar maps
					if (estd::has(voters_.immutables_, const_str))
					{
						cands.emplace(
							Symbol{CAND_TYPE::SCALAR, const_str}, CtxsT{});
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
	void visit (teq::iFunctor* func) override
	{
		if (false == estd::has(candidates_, func))
		{
			auto& children = func->get_children();
			for (auto& child : children)
			{
				child.get_tensor()->accept(*this);
			}

			if (std::all_of(children.begin(), children.end(),
				[this](const teq::FuncArg& child) -> bool
				{
					auto ctens = child.get_tensor().get();
					return estd::has(this->candidates_[ctens],
						Symbol{CAND_TYPE::CONST, ctens->to_string()});
				}))
			{
				// all children are constants
				// therefore mark this as constant
				std::string const_str = func->to_string();
				candidates_.emplace(func, CandsT{
					{Symbol{CAND_TYPE::CONST, const_str}, CtxsT{}},
				});

				// mark as scalar if func's children are scalar
				// in order to propagate scalar info to parents
				if (scalarize_)
				{
					if (std::all_of(children.begin(), children.end(),
						[this](const teq::FuncArg& child) -> bool
						{
							auto ctens = child.get_tensor().get();
							std::string scalar_str = scalarize_(ctens);
							return estd::has(this->candidates_[ctens],
								Symbol{CAND_TYPE::SCALAR, scalar_str});
						}))
					{
						std::string scalar_str = scalarize_(func);
						candidates_[func].emplace(
							Symbol{CAND_TYPE::SCALAR, scalar_str}, CtxsT{});
					}
				}
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
			tag::SubgraphsT sgs;
			if (estd::get(sgs, group_head_, func))
			{
				// look for candidates in each of the potential subgraphs
				for (tag::SgraphptrT sg : sgs)
				{
					auto bit = voters_.branches_.find(
						group_prefix + sg->group_);
					if (voters_.branches_.end() != bit)
					{
						// todo: store sg->children_ as teq::ArgsT
						CandArgsT args;
						args.reserve(children.size());
						for (auto& sgcpair : sg->children_)
						{
							auto ctens = sgcpair.second;
							args.push_back(CandArg{
								ctens,
								candidates_[sgcpair.first],
								teq::identity,
								teq::CoordptrT(),
							});
						}
						CandsT group_cands = bit->second->inspect(args);
						out_cands.insert(
							group_cands.begin(), group_cands.end());
					}
				}
			}

			candidates_.emplace(func, out_cands);
		}
	}

	/// Conversion voters to identify candidates
	VoterPool voters_;

	/// Map real TEQ tensors to candidates identified as TEQ graph is visited
	std::unordered_map<teq::iTensor*,CandsT> candidates_;

	/// Root of grouped subgraphs
	tag::SubgraphAssocsT group_head_;

	/// Function that returns constant representation of tensor
	std::function<std::string(teq::iTensor*)> scalarize_;
};

}

#endif // OPT_MATCHER_HPP
