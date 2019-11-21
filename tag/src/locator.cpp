#include "tag/locator.hpp"

#ifdef TAG_LOCATOR_HPP

namespace tag
{

std::string display_location (teq::TensptrT tens,
	teq::TensptrsT known_roots, TagRegistry& tagreg)
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
	if (known_roots.size() > 0)
	{
		teq::ParentFinder pfinder;
		for (auto root : known_roots)
		{
			root->accept(pfinder);
		}
		auto& parents = pfinder.parents_[tens.get()];
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
	tag::TagRepsT tags = tagreg.get_tags(tens.get());
	if (tags.size() > 0)
	{
		auto it = tags.begin(), et = tags.end();
		out << it->first << ":"
			<< fmts::to_string(it->second.begin(), it->second.end());
		for (++it; it != et; ++it)
		{
			out << "," << it->first << ":"
				<< fmts::to_string(it->second.begin(), it->second.end());
		}
	}
	out << "}";
	if (teq::iFunctor* f = dynamic_cast<teq::iFunctor*>(tens.get()))
	{
		auto args = f->get_children();
		for (const teq::iEdge& arg : args)
		{
			auto argtens = arg.get_tensor();
			out << "\n" << branchfmt
				<< argtens->to_string() << ":"
				<< argtens->shape().to_string();
		}
	}
	return out.str();
}

}

#endif
