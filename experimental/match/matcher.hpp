#include "ade/traveler.hpp"

#include "experimental/match/voter.hpp"

#ifndef OPT_MATCH_MATCHER_HPP
#define OPT_MATCH_MATCHER_HPP

namespace opt
{

namespace match
{

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
    Matcher (const VoterPool& voters) : voters_(&voters) {}

    /// Implementation of iTraveler
    void visit (ade::iLeaf* leaf) override
    {
        if (false == util::has(candidates_, leaf))
        {
            if (tag::has_property(leaf, tag::immutable_tag) &&
                is_scalar(leaf))
            {
                // match against scalar maps
                std::string scalar_str = leaf->to_string();
                auto it = voters_->immutables_.find(scalar_str);
                if (voters_->immutables_.end() != it)
                {
                    candidates_.emplace(leaf, CandsT{
                        {Symbol{SCALAR, scalar_str}, {}},
                    });
                    return;
                }
            }
            candidates_.emplace(leaf, CandsT{});
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

            // functor/group
            std::string opname = func->get_opcode().name_;
            auto it = voters_->branches_.find(opname);
            if (voters_->branches_.end() != it)
            {
                CandArgsT args;
                args.reserve(children.size());
                for (auto& child : children)
                {
                    auto ctens = child.get_tensor().get();
                    args.push_back(CandArg{
                        ctens,
                        candidates_[ctens],
                        child.get_shaper(),
                        child.get_coorder(),
                    });
                }
                candidates_.emplace(func, it->second->inspect(args));
            }
            else
            {
                candidates_.emplace(func, CandsT{});
            }
        }
    }

    // created by parser
    const VoterPool* voters_;

    // generated as visited
    std::unordered_map<ade::iTensor*,CandsT> candidates_;
};

}

}

#endif // OPT_MATCH_MATCHER_HPP
