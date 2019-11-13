///
/// optimize.hpp
/// opt
///
/// Purpose:
/// Implement algorithm that applies conversion rules to graph roots
///

#include "tag/prop.hpp"

#include "opt/matcher.hpp"

#ifndef OPT_OPTIMIZE_HPP
#define OPT_OPTIMIZE_HPP

namespace opt
{

struct iTarget
{
	virtual ~iTarget (void) = default;

	virtual teq::TensptrT convert (
		teq::Shape outshape, const Candidate& candidate) const = 0;
};

using TargptrT = std::shared_ptr<iTarget>;

/// Encapsulation of all conversion rules
struct CversionCtx
{
	/// Return function id (fid) of promising matcher once found
	teq::TensptrT optimize (MatchCtxT& out, teq::iFunctor* f) const
	{
		const teq::CEdgesT& args = f->get_children();
		std::string opname = f->get_opcode().name_;
		if (estd::has(matchers_, opname))
		{
			const MatchsT& matchers = matchers_.at(opname);
			for (auto& matcher : matchers)
			{
				auto fid = matcher->get_fid();
				auto cands = matcher->match(out, args);
				if (cands.size() > 0)
				{
					auto& matched_map = out[f];
					matched_map[fid] = cands;
					if (estd::has(targets_, fid))
					{
						if (cands.size() > 1)
						{
							logs::warn("ambiguous matcher results: more than 1 candidate");
							// todo: dump these candidates
						}
						return targets_.at(fid)->convert(f->shape(), cands[0]);
					}
				}
			}
		}
		return nullptr;
	}

	/// Matching opname to Matchers
	std::unordered_map<std::string,MatchsT> matchers_;

	/// Matches root matcher fid to target
	std::unordered_map<std::string,TargptrT>  targets_;
};

using ParentReplF = std::function<void(teq::TensptrT,teq::iTensor*)>;

using PerFuncFiltF = std::function<teq::TensptrT(teq::FuncptrT&,ParentReplF)>;

using FilterF = std::function<void(teq::TensptrsT&)>;

struct CustomFilters final
{
	// custom filters to run against each tensor node before applying rules
	std::vector<PerFuncFiltF> prenode_filters_;

	// custom filters to run against each tensor node after applying rules
	std::vector<PerFuncFiltF> postnode_filters_;

	// custom filters to run against entire graph before applying rules
	std::vector<FilterF> prefilters_;

	// custom filters to run against entire graph after applying rules
	std::vector<FilterF> postfilters_;
};

using CversionsT = std::vector<std::pair<teq::FuncptrT,teq::TensptrT>>;

/// Replace source tensor's position with target's position
/// in the sense that all parents of source (found in pfinder)
/// take on target as the new child in place of source's
void replace_parents (const teq::ParentFinder& pfinder,
	teq::TensptrT target, teq::iTensor* source,
	tag::TagRegistry& registry = tag::get_reg());

/// Return optimized roots where optimization rules are applied to subgraphs
/// Optimized graph roots are moved back to their corresponding root tensors
/// Additionally two or more tensors sharing symbolically identical
/// representations are "joined" (with the exception of tensors in roots set)
teq::TensptrsT optimize (teq::TensptrsT roots,
	const CversionCtx& opts, const CustomFilters& filters);

}

#endif // OPT_OPTIMIZE_HPP
