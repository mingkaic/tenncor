
#include <list>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "opt/ematcher.hpp"

#ifndef OPT_MATCHER_HPP
#define OPT_MATCHER_HPP

namespace opt
{

void merge_cands (CandsT& out, const CandsT& a, const CandsT& b);

static boost::uuids::random_generator uuid_gen;

struct iMatcher
{
	virtual ~iMatcher (void) = default;

	virtual CandsT match (const MatchCtxT& ctx, const teq::CEdgesT& args) const = 0;

	virtual std::string get_fid (void) const = 0;

	virtual void add_edge (ScalarEMatcher* matcher) = 0;

	virtual void add_edge (AnyEMatcher* matcher) = 0;

	virtual void add_edge (FuncEMatcher* matcher) = 0;
};

struct OrderedMatcher final : public iMatcher
{
	OrderedMatcher (std::string variadic) : variadic_(variadic) {}

	CandsT match (const MatchCtxT& ctx, const teq::CEdgesT& args) const override
	{
		size_t nematchers = edges_.size();
		size_t nargs = args.size();
		if ((nargs != nematchers && variadic_.empty()) ||
			nematchers > nargs)
		{
			// nematchers > nargs = more required matches than args available
			// args is impossible to match edge matchers
			return CandsT{};
		}
		// assert nematchers <= nargs

		CandsT cands;
		for (size_t i = 0; i < nematchers; ++i)
		{
			CandsT ecands = edges_[i]->match(ctx, args[i]);
			if (ecands.empty())
			{
				// no candidates found
				return ecands;
			}
			CandsT next_cands;
			merge_cands(next_cands, cands, ecands);
			cands = next_cands;
		}
		if (variadic_.size() > 0)
		{
			teq::CEdgesT varis(args.begin() + nematchers, args.end());
			for (auto& cand : cands)
			{
				cand.variadic_[variadic_] = varis;
			}
		}
		return cands;
	}

	std::string get_fid (void) const override
	{
		return id_;
	}

	void add_edge (ScalarEMatcher* matcher) override
	{
		edges_.emplace(edges_.end(), EMatchptrT(matcher));
	}

	void add_edge (AnyEMatcher* matcher) override
	{
		edges_.emplace(edges_.end(), EMatchptrT(matcher));
	}

	void add_edge (FuncEMatcher* matcher) override
	{
		edges_.emplace(edges_.end(), EMatchptrT(matcher));
	}

private:
	const std::string id_ = boost::uuids::to_string(uuid_gen());

	std::string variadic_;

	EMatchptrsT edges_;
};

struct CommutativeMatcher final : public iMatcher
{
	CommutativeMatcher (std::string variadic) : variadic_(variadic) {}

	/// Return matched candidates, empty candidates indicate no matches
	CandsT match (const MatchCtxT& ctx, const teq::CEdgesT& args) const override
	{
		size_t nematchers = scalars_.size() + funcs_.size() + anys_.size();
		size_t nargs = args.size();
		if ((nargs != nematchers && variadic_.empty()) ||
			nematchers > nargs)
		{
			// nematchers > nargs = more required matches than args available
			// args is impossible to match edge matchers
			return CandsT{};
		}
		// assert nematchers <= nargs

		CandsT cands;
		std::list<std::reference_wrapper<const teq::iEdge>> unmatched(
			args.begin(), args.end());
		// match scalars first
		for (auto& smatcher : scalars_)
		{
			for (auto it = unmatched.begin(), et = unmatched.end();
				it != et; ++it)
			{
				CandsT ecands = smatcher->match(ctx, *it);
				if (ecands.size() > 0)
				{
					// candidates found
					unmatched.erase(it);
					CandsT next_cands;
					merge_cands(next_cands, cands, ecands);
					cands = next_cands;
					break;
				}
			}
		}
		if (scalars_.size() > 0 && cands.empty())
		{
			// failed to match scalars
			return CandsT{};
		}

		// match functors next
		for (auto& fmatcher : funcs_)
		{
			for (auto it = unmatched.begin(), et = unmatched.end();
				it != et; ++it)
			{
				CandsT ecands = fmatcher->match(ctx, *it);
				if (ecands.size() > 0)
				{
					// candidates found
					unmatched.erase(it);
					CandsT next_cands;
					merge_cands(next_cands, cands, ecands);
					cands = next_cands;
					break;
				}
			}
		}
		if (funcs_.size() > 0 && cands.empty())
		{
			// failed to match scalars
			return CandsT{};
		}
		// assert not variadic -> anys_.size() == unmatched.size() &&
		//	variadic -> anys_.size() <= unmatched.size()

		// match remaining anys
		teq::CEdgesT remaining(unmatched.begin(), unmatched.end());
		size_t nremaining = remaining.size();
		std::vector<size_t> indices(nremaining);
		std::iota(indices.begin(), indices.end(), 0);
		size_t nanys = anys_.size();
		CandsT out;
		do
		{
			// select first nanys indices,
			// and dump remaining as variadic
			CandsT local_cands = cands;
			for (size_t i = 0; i < nanys && local_cands.size() > 0; ++i)
			{
				auto ecands = anys_[i]->match(ctx, remaining[indices[i]]);
				if (ecands.empty())
				{
					local_cands.clear();
				}
				else
				{
					CandsT next_cands;
					merge_cands(next_cands, local_cands, ecands);
					local_cands = next_cands;
				}
			}

			if (local_cands.empty())
			{
				continue;
			}

			// dump remaining[indices[nanys]:] as variadic
			teq::CEdgesT varis;
			for (size_t i = nanys; i < nremaining; ++i)
			{
				varis.push_back(remaining[indices[i]]);
			}
			for (auto& local_cand : local_cands)
			{
				local_cand.variadic_.emplace(variadic_, varis);
			}
			out.insert(out.end(), local_cands.begin(), local_cands.end());
		}
		while (std::next_permutation(indices.begin(), indices.end()));

		return out;
	}

	std::string get_fid (void) const override
	{
		return id_;
	}

	void add_edge (ScalarEMatcher* matcher) override
	{
		scalars_.emplace(scalars_.end(),
			std::unique_ptr<ScalarEMatcher>(matcher));
	}

	void add_edge (AnyEMatcher* matcher) override
	{
		anys_.emplace(anys_.end(), std::unique_ptr<AnyEMatcher>(matcher));
	}

	void add_edge (FuncEMatcher* matcher) override
	{
		funcs_.emplace(funcs_.end(), std::unique_ptr<FuncEMatcher>(matcher));
	}

private:
	const std::string id_ = boost::uuids::to_string(uuid_gen());

	std::string variadic_;

	std::vector<std::unique_ptr<ScalarEMatcher>> scalars_;

	std::vector<std::unique_ptr<FuncEMatcher>> funcs_;

	std::vector<std::unique_ptr<AnyEMatcher>> anys_;
};

using MatchptrT = std::shared_ptr<iMatcher>;

using MatchsT = std::vector<MatchptrT>;

}

#endif // OPT_MATCHER_HPP
