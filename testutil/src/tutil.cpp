#include "diff/msg.hpp"

#include "testutil/tutil.hpp"

#ifdef TEST_TUTIL_HPP

namespace tutil
{

std::string compare_graph (std::istream& expectstr, teq::iTensor* root,
	bool showshape, LabelsMapT labels)
{
	PrettyEquation artist;
	artist.showshape_ = showshape;
	artist.labels_ = labels;
	std::stringstream gotstr;
	artist.print(gotstr, root);
	std::vector<std::string> expects;
	std::vector<std::string> gots;
	std::string line;
	while (std::getline(expectstr, line))
	{
		fmts::trim(line);
		if (line.size() > 0)
		{
			expects.push_back(line);
		}
	}
	while (std::getline(gotstr, line))
	{
		fmts::trim(line);
		if (line.size() > 0)
		{
			gots.push_back(line);
		}
	}
	return diff::diff_msg(expects, gots);
}

std::string compare_graph (std::istream& expectstr, teq::TensptrT root,
	bool showshape, LabelsMapT labels)
{
	return compare_graph(expectstr, root.get(), showshape, labels);
}

}

#endif
