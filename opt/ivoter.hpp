///
/// ivoter.hpp
/// opt
///
/// Purpose:
/// Define rule voter to identify graph nodes
///

extern "C" {
#include "opt/parse/def.h"
}

#include "opt/stats.hpp"
#include "opt/candidate.hpp"

#ifndef OPT_IVOTER_HPP
#define OPT_IVOTER_HPP

namespace opt
{

/// Argument voter for functors
struct VoterArg final
{
	VoterArg (std::string label,
		teq::CoordptrT shaper,
		teq::CoordptrT coorder,
		SUBGRAPH_TYPE type) :
		label_(label),
		shaper_(shaper),
		coorder_(coorder),
		type_(type) {}

	/// Return true if arg matches this only add to ctxs if matches
	bool match (CtxsT& ctxs, const CandArg& arg) const
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
				return estd::has(arg.candidates_, Symbol{
					CAND_TYPE::SCALAR,
					label_,
				});
			case ::SUBGRAPH_TYPE::BRANCH:
			{
				// look for intermediate candidate in arg.candidates
				auto it = arg.candidates_.find(Symbol{
					CAND_TYPE::INTERM,
					label_,
				});
				if (arg.candidates_.end() == it)
				{
					return false;
				}
				auto& cand_ctxs = it->second;
				CtxsT matching_ctxs;
				if (ctxs.empty())
				{
					matching_ctxs = cand_ctxs;
				}
				for (const ContexT& ctx : ctxs)
				{
					for (ContexT cand_tx : cand_ctxs)
					{
						// if the cand_tx map and ctx have a
						// non-conflicting intersection, mark as matched
						if (std::all_of(ctx.begin(), ctx.end(),
							[&](const std::pair<std::string,CtxValT>& ctxpair)
							{
								auto ait = cand_tx.find(ctxpair.first);
								if (cand_tx.end() == ait)
								{
									return true;
								}
								auto& ctens = ctxpair.second;
								auto& cand_ctens = ait->second;
								return std::equal(ctens.begin(), ctens.end(),
									cand_ctens.begin());
							}))
						{
							// merge
							cand_tx.insert(ctx.begin(), ctx.end());
							matching_ctxs.emplace(cand_tx);
						}
					}
				}
				bool has_match = matching_ctxs.size() > 0;
				if (has_match)
				{
					ctxs = matching_ctxs;
				}
				return has_match;
			}
			case ::SUBGRAPH_TYPE::ANY:
			{
				CtxsT matching_ctxs;
				if (ctxs.empty())
				{
					matching_ctxs.emplace(ContexT{{label_, {arg.tensor_}}});
				}
				for (ContexT ctx : ctxs)
				{
					auto it = ctx.find(label_);
					if (ctx.end() == it || estd::has(it->second, arg.tensor_))
					{
						// matching ANY
						ctx.emplace(label_, CtxValT{arg.tensor_});
						matching_ctxs.emplace(ctx);
					}
				}
				bool has_match = matching_ctxs.size() > 0;
				if (has_match)
				{
					ctxs = matching_ctxs;
				}
				return has_match;
			}
		}
		return true;
	}

	std::string label_;

	teq::CoordptrT shaper_;

	teq::CoordptrT coorder_;

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

void sort_vargs (VoterArgsT& args);

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
	return lhs.size() == rhs.size() &&
		std::equal(lhs.begin(), lhs.end(), rhs.begin(),
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

using VotptrT = std::shared_ptr<iVoter>;

struct VoterPool
{
	std::unordered_set<std::string> immutables_;

	std::unordered_map<std::string,VotptrT> branches_;
};

}

#endif // OPT_IVOTER_HPP
