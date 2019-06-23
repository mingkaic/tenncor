#include "opt/ivoter.hpp"

#ifndef OPT_VOTER_HPP
#define OPT_VOTER_HPP

namespace opt
{

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
			CtxsT ctxs;
			const VoterArgsT& vargs = vpair.first;
			if (vargs.size() != args.size())
			{
				continue;
			}
			if ([&]() -> bool
				{
					for (size_t i = 0, n = args.size(); i < n; ++i)
					{
						if (false == vargs[i].match(ctxs, args[i]))
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
			out[vpair.second].insert(ctxs.begin(), ctxs.end());
		}
		return out;
	}

	std::string label_;

	std::unordered_map<VoterArgsT,Symbol,OrdrHasher> args_;
};

// todo: ensure comm voters are inspected after all available ordr voters are inspected (optimization)
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
			CtxsT ctxs;
			std::list<CandArg> unmatched(args.begin(), args.end());
			// attempt matching scalars first
			bool match_failed = false;
			for (const VoterArg& sarg : vargs.scalars_)
			{
				match_failed = true;
				for (auto it = unmatched.begin(), et = unmatched.end();
					it != et; ++it)
				{
					if (sarg.match(ctxs, *it))
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
					if (barg.match(ctxs, *it))
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

			// create permutations of remaining against anys matches
			std::vector<CandArg> remaining(unmatched.begin(), unmatched.end());
			size_t nremaining = remaining.size();
			std::vector<size_t> indices(nremaining);
			std::iota(indices.begin(), indices.end(), 0);
			do
			{
				bool matched = true;
				CtxsT local_ctxs = ctxs;
				for (size_t i = 0; i < nremaining && matched; ++i)
				{
					matched = vargs.anys_[i].match(local_ctxs,
						remaining[indices[i]]);
				}
				if (false == matched)
				{
					break;
				}
				out[vpair.second].insert(local_ctxs.begin(), local_ctxs.end());
			}
			while (std::next_permutation(indices.begin(), indices.end()));
		}
		return out;
	}

	std::string label_;

	std::unordered_map<SegVArgs,Symbol,CommHasher> args_;
};

struct VariadicVoter final : public iVoter
{
	VariadicVoter (std::string label, std::string variadic) :
		label_(label), variadic_(variadic) {}

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
			if (vargs.size() > args.size())
			{
				// not enough voter arguments to match candidate arguments
				continue;
			}
			CtxsT ctxs;
			std::list<CandArg> unmatched(args.begin(), args.end());
			// attempt matching scalars first
			bool match_failed = false;
			for (const VoterArg& sarg : vargs.scalars_)
			{
				match_failed = true;
				for (auto it = unmatched.begin(), et = unmatched.end();
					it != et; ++it)
				{
					if (sarg.match(ctxs, *it))
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
					if (barg.match(ctxs, *it))
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

			// create permutations of remaining against anys matches
			std::vector<CandArg> remaining(unmatched.begin(), unmatched.end());
			size_t nremaining = remaining.size();
			std::vector<size_t> indices(nremaining);
			std::iota(indices.begin(), indices.end(), 0);

			size_t nanys = vargs.anys_.size();
			do
			{
				// select first nanys indices,
				// and dump remaining as variadic
				bool matched = true;
				CtxsT local_ctxs = ctxs;
				for (size_t i = 0; i < nanys && matched; ++i)
				{
					matched = vargs.anys_[i].match(local_ctxs,
						remaining[indices[i]]);
				}
				if (false == matched)
				{
					break;
				}
				CtxValT cvals;
				for (size_t i = nanys; i < nremaining; ++i)
				{
					// dump remaining[indices[i]] as variadic
					cvals.emplace(remaining[indices[i]].tensor_); // todo: also store coorder and shaper
				}
				CtxsT& out_ctxs = out[vpair.second];
				for (ContexT ctx : local_ctxs)
				{
					ctx.emplace(variadic_, cvals);
					out_ctxs.emplace(ctx);
				}
			}
			while (std::next_permutation(indices.begin(), indices.end()));
		}
		return out;
	}

	std::string label_;

	std::string variadic_;

	std::unordered_map<SegVArgs,Symbol,CommHasher> args_;
};

}

#endif // OPT_VOTER_HPP
