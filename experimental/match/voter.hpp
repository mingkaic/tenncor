extern "C" {
#include "experimental/match/parse/def.h"
}

#include "experimental/match/stats.hpp"
#include "experimental/match/candidate.hpp"

#ifndef OPT_MATCH_VOTER_HPP
#define OPT_MATCH_VOTER_HPP

namespace opt
{

namespace match
{

struct VoterArg final
{
    VoterArg (std::string label,
        ade::CoordptrT shaper,
        ade::CoordptrT coorder,
        SUBGRAPH_TYPE type) :
        label_(label),
        shaper_(shaper),
        coorder_(coorder),
        type_(type) {}

    // return true if arg matches this
    // only add to anys if matches
    bool match (AnyMapT& anys, const CandArg& arg) const
    {
        // match arg.shaper_ and arg.coorder_
        if (false == is_equal(arg.shaper_, shaper_) ||
            false == is_equal(arg.coorder_, coorder_))
        {
            return false;
        }
        switch (type_)
        {
            case ::SUBGRAPH_TYPE::SCALAR:
                // look for scalar candidate in arg.candidates
                return util::has(arg.candidates_, Symbol{SCALAR, label_});
            case ::SUBGRAPH_TYPE::BRANCH:
            {
                // look for intermediate candidate in arg.candidates
                auto it = arg.candidates_.find(Symbol{INTERM, label_});
                if (arg.candidates_.end() == it)
                {
                    return false;
                }
                auto& canys = it->second;
                for (const AnyMapT& cany : canys)
                // this is a probabilistic approach. only takes first matching candidate any (todo: consider change)
                {
                    // if the cany map and anys have a
                    // non-conflicting intersection, mark as matched
                    if (std::all_of(anys.begin(), anys.end(),
                        [&](const std::pair<std::string,
                            ade::iTensor*>& anypair)
                        {
                            auto ait = cany.find(anypair.first);
                            return cany.end() == ait ||
                                anypair.second == ait->second;
                        }))
                    {
                        // merge
                        anys.insert(cany.begin(), cany.end());
                        return true;
                    }
                }
                // failed to find one matching any map
                return false;
            }
            case ::SUBGRAPH_TYPE::ANY:
            {
                auto it = anys.find(label_);
                if (anys.end() == it)
                {
                    anys.emplace(label_, arg.tensor_);
                }
                else if (it->second != arg.tensor_)
                {
                    // mismatching ANY
                    return false;
                }
            }
        }
        return true;
    }

    std::string label_;

    ade::CoordptrT shaper_;

    ade::CoordptrT coorder_;

    SUBGRAPH_TYPE type_;
};

using VoterArgsT = std::vector<VoterArg>;

struct SegVArgs
{
    size_t size (void) const
    {
        return scalars_.size() + branches_.size() + anys_.size();
    }

    VoterArgsT scalars_;

    VoterArgsT branches_;

    VoterArgsT anys_;
};

void sort_vargs (VoterArgsT& args)
{
    // sort args
    std::sort(args.begin(), args.end(),
        [](const VoterArg& a, const VoterArg& b)
        {
            if (a.label_ == b.label_)
            {
                if (is_equal(a.shaper_, b.shaper_))
                {
                    return lt(a.coorder_, b.coorder_);
                }
                return lt(a.shaper_, b.shaper_);
            }
            return a.label_ < b.label_;
        });
}

struct OrdrHasher final
{
    size_t operator() (const VoterArgsT& args) const
    {
        size_t seed = 0;
        hash_combine(seed, args);
        return seed;
    }

    void hash_combine (size_t& seed, const VoterArgsT& args) const
    {
        for (const VoterArg& arg : args)
        {
            std::tuple<std::string,std::string,std::string,size_t>
            hash_target = {
                arg.label_,
                to_string(arg.shaper_),
                to_string(arg.coorder_),
                arg.type_,
            };
            boost::hash_combine(seed, hash_target);
        }
    }
};

inline bool operator == (const VoterArgsT& lhs, const VoterArgsT& rhs)
{
    return std::equal(lhs.begin(), lhs.end(), rhs.begin(),
        [](const VoterArg& l, const VoterArg& r)
        {
            return l.label_ == r.label_ &&
                is_equal(l.shaper_, r.shaper_) &&
                is_equal(l.coorder_, r.coorder_) &&
                l.type_ == r.type_;
        });
}

struct CommHasher final
{
    size_t operator() (const SegVArgs& args) const
    {
        size_t seed = 0;
        hasher_.hash_combine(seed, args.scalars_);
        hasher_.hash_combine(seed, args.branches_);
        hasher_.hash_combine(seed, args.anys_);
        return seed;
    }

    OrdrHasher hasher_;
};

inline bool operator == (const SegVArgs& lhs, const SegVArgs& rhs)
{
	return lhs.scalars_ == rhs.scalars_ &&
        lhs.branches_ == rhs.branches_ &&
        lhs.anys_ == rhs.anys_;
}

// select candidates
struct iVoter
{
    virtual ~iVoter (void) = default;

    virtual void emplace (VoterArgsT args, Symbol cand) = 0;

    virtual CandsT inspect (const CandArgsT& args) const = 0;
};

using VotptrT = std::unique_ptr<iVoter>;

struct VoterPool
{
    std::unordered_map<std::string,::Subgraph*> converts_;

    std::unordered_set<std::string> immutables_;

    std::unordered_map<std::string,VotptrT> branches_;
};

struct OrdrVoter final : public iVoter
{
    OrdrVoter (std::string label) : label_(label) {}

    void emplace (VoterArgsT args, Symbol sym) override
    {
        args_.emplace(args, sym);
    }

    CandsT inspect (const CandArgsT& args) const override
    {
        CandsT out;
        out.reserve(args_.size());
        for (const auto& vpair : args_)
        {
            AnyMapT anys;
            const VoterArgsT& vargs = vpair.first;
            if (vargs.size() != args.size())
            {
                continue;
            }
            if ([&]() -> bool
                {
                    for (size_t i = 0, n = args.size(); i < n; ++i)
                    {
                        if (false == vargs[i].match(anys, args[i]))
                        {
                            return true;
                        }
                    }
                    return false;
                }())
            {
                // failure to match one of the arguments
                continue;
            }
            out[vpair.second].push_back(anys);
        }
        return out;
    }

    std::string label_;

    std::unordered_map<VoterArgsT,Symbol,OrdrHasher> args_;
};

// todo: ensure comm voters are inspected after all available ordr voters are inspected
struct CommVoter final : public iVoter
{
    CommVoter (std::string label) : label_(label) {}

    void emplace (VoterArgsT args, Symbol sym) override
    {
        // sort args
        SegVArgs segs;
        for (VoterArg& arg : args)
        {
            switch (arg.type_)
            {
                case ::SUBGRAPH_TYPE::SCALAR:
                    segs.scalars_.push_back(arg);
                    break;
                case ::SUBGRAPH_TYPE::BRANCH:
                    segs.branches_.push_back(arg);
                    break;
                case ::SUBGRAPH_TYPE::ANY:
                default:
                    segs.anys_.push_back(arg);
            }
        }
        if (segs.branches_.size() > 1)
        {
            logs::fatal("implementation limit: "
                "cannot have more than 1 operator as an argument of the "
                "commutative operator for the source subgraph");
        }
        sort_vargs(segs.scalars_);
        sort_vargs(segs.anys_);
        args_.emplace(segs, sym);
    }

    CandsT inspect (const CandArgsT& args) const override
    {
        CandsT out;
        out.reserve(args_.size());
        for (const auto& vpair : args_)
        {
            const SegVArgs& vargs = vpair.first;
            if (vargs.size() != args.size())
            {
                continue;
            }
            AnyMapT anys;
            std::list<CandArg> unmatched(args.begin(), args.end());
            // attempt matching scalars first
            bool match_failed = false;
            for (const VoterArg& sarg : vargs.scalars_)
            {
                match_failed = true;
                for (auto it = unmatched.begin(), et = unmatched.end();
                    it != et; ++it)
                {
                    if (sarg.match(anys, *it))
                    {
                        unmatched.erase(it);
                        match_failed = false;
                        break;
                    }
                }
                if (match_failed)
                {
                    break;
                }
            }
            // none of the unmatched args matched
            // a scalar voter argument
            if (match_failed)
            {
                continue;
            }

            for (const VoterArg& barg : vargs.branches_)
            {
                match_failed = true;
                for (auto it = unmatched.begin(), et = unmatched.end();
                    it != et; ++it)
                {
                    if (barg.match(anys, *it))
                    {
                        unmatched.erase(it);
                        match_failed = false;
                        break;
                    }
                }
                if (match_failed)
                {
                    break;
                }
            }
            // none of the unmatched args matched
            // a branch voter argument
            if (match_failed)
            {
                continue;
            }

            VoterArgsT engaged;
            VoterArgsT liberal;
            for (const VoterArg& aarg : vargs.anys_)
            {
                if (util::has(anys, aarg.label_))
                {
                    engaged.push_back(aarg);
                }
                else
                {
                    liberal.push_back(aarg);
                }
            }
            for (const VoterArg& aarg : engaged)
            {
                match_failed = true;
                for (auto it = unmatched.begin(), et = unmatched.end();
                    it != et; ++it)
                {
                    if (aarg.match(anys, *it))
                    {
                        unmatched.erase(it);
                        match_failed = false;
                        break;
                    }
                }
                if (match_failed)
                {
                    break;
                }
            }
            // none of the unmatched args matched
            // one of the engaged any voter argument
            if (match_failed)
            {
                continue;
            }

            // create permutations of liberal matches
            std::vector<CandArg> remaining(unmatched.begin(), unmatched.end());
            size_t nremaining = remaining.size();
            std::vector<size_t> indices(nremaining);
            std::iota(indices.begin(), indices.end(), 0);
            do
            {
                bool matched = true;
                AnyMapT local_any = anys;
                for (size_t i = 0; i < nremaining && matched; ++i)
                {
                    // todo: this opens too much ambiguity, consider limiting
                    matched = liberal[i].match(local_any,
                        remaining[indices[i]]);
                }
                if (false == matched)
                {
                    break;
                }
                out[vpair.second].push_back(local_any);
            }
            while (std::next_permutation(indices.begin(), indices.end()));
        }
        return out;
    }

    std::string label_;

    std::unordered_map<SegVArgs,Symbol,CommHasher> args_;
};

}

}

#endif // OPT_MATCH_VOTER_HPP
