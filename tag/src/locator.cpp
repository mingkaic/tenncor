#include "tag/locator.hpp"

#ifdef TAG_LOCATOR_HPP

namespace tag
{

std::string display_location (teq::TensptrT tens,
	teq::TensptrsT known_parents, TagRegistry& tagreg)
{
	if (nullptr == tens)
	{
		return "<nil>";
	}
	// fmt:
	// <parents>
	//  `--<rep> {<tagtype>: <tagvals>,...}
	//      `--<childrep>
	//       ...
	std::stringstream out;
	std::string branchfmt = " `--";
	if (known_parents.size() > 0)
	{
		out << known_parents[0]->to_string() << ":"
			<< known_parents[0]->shape().to_string();
		for (size_t i = 1, n = known_parents.size(); i < n; ++i)
		{
			teq::TensptrT& parent = known_parents[i];
			out << parent->to_string() << ":"
				<< parent->shape().to_string();
		}
		out << "\n" << branchfmt;
		branchfmt = "    " + branchfmt;
	}
	out << tens->to_string() << ":" << tens->shape().to_string() << "{";
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
