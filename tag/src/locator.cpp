#include "tag/locator.hpp"

#ifdef TAG_LOCATOR_HPP

namespace tag
{

std::string display_location (teq::iTensor* tens,
	teq::TensT known_roots, TagRegistry& tagreg)
{
	if (nullptr == tens)
	{
		return "<nil>";
	}
	// format:
	// <parents>
	//  `--<rep> {<tagtype>: <tagvals>,...}
	//      `--<childrep>
	//       ...
	std::stringstream out;
	std::string branchfmt = " `--";
	teq::TensT roots;
	std::copy_if(known_roots.begin(), known_roots.end(),
		std::back_inserter(roots),
		[](teq::iTensor* root) { return root != nullptr; });
	if (roots.size() > 0)
	{
		teq::ParentFinder pfinder;
		for (auto root : roots)
		{
			root->accept(pfinder);
		}
		auto& parents = pfinder.parents_[tens];
		size_t n = parents.size();
		std::vector<std::string> plabels;
		plabels.reserve(n);
		for (auto& p : parents)
		{
			auto parent = p.first;
			plabels.push_back(
				parent->to_string() + ":" +
				parent->shape().to_string());
		}
		if (n > 0)
		{
			std::sort(plabels.begin(), plabels.end());
			out << fmts::join("\n", plabels.begin(), plabels.end())
				<< "\n" << branchfmt;
			branchfmt = "    " + branchfmt;
		}
	}
	out << ">" << tens->to_string() << "<:" << tens->shape().to_string() << "{";
	tag::TagRepsT tags = tagreg.get_tags(tens);
	size_t ntags = tags.size();
	if (ntags > 0)
	{
		std::vector<std::string> tagstr;
		tagstr.reserve(ntags);
		std::transform(tags.begin(), tags.end(), std::back_inserter(tagstr),
			[](const std::pair<std::string,std::vector<std::string>>& tpair)
			{
				return tpair.first + ":" + fmts::to_string(
					tpair.second.begin(), tpair.second.end());
			});
		out << fmts::join(",", tagstr.begin(), tagstr.end());
	}
	out << "}";
	if (teq::iFunctor* f = dynamic_cast<teq::iFunctor*>(tens))
	{
		auto args = f->get_children();
		for (const teq::iEdge& arg : args)
		{
			auto argtens = arg.get_tensor();
			out << "\n" << branchfmt << "("
				<< argtens->to_string() << "):"
				<< argtens->shape().to_string();
		}
	}
	return out.str();
}

std::string display_location (teq::TensptrT tens,
	teq::TensptrsT known_roots, TagRegistry& tagreg)
{
	teq::TensT roots;
	std::transform(known_roots.begin(), known_roots.end(),
		std::back_inserter(roots),
		[](teq::TensptrT root) { return root.get(); });
	return display_location(tens.get(), roots, tagreg);
}

}

#endif
