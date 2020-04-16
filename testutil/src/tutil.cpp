#include "diff/msg.hpp"

#include "testutil/tutil.hpp"

#ifdef TEST_TUTIL_HPP

namespace tutil
{

const size_t peek_limit = 256;

std::string compare_graph (std::istream& expectstr, teq::iTensor* root,
	bool showshape, LabelsMapT labels)
{
	PrettyEquation artist;
	artist.showshape_ = showshape;
	artist.labels_ = labels;
	std::stringstream gotstr;
	artist.print(gotstr, root);
	std::string out;
	bool exrun, gorun;
	exrun = gorun = true;
	while (out.empty() && (exrun || gorun))
	{
		std::vector<std::string> expects;
		std::vector<std::string> gots;
		std::string line;
		for (size_t i = 0; i < peek_limit && exrun; ++i)
		{
			exrun = (bool) std::getline(expectstr, line);
			if (exrun)
			{
				fmts::trim(line);
				if (line.size() > 0)
				{
					expects.push_back(line);
				}
			}
		}
		for (size_t i = 0; i < peek_limit && gorun; ++i)
		{
			gorun = (bool) std::getline(gotstr, line);
			if (gorun)
			{
				fmts::trim(line);
				if (line.size() > 0)
				{
					gots.push_back(line);
				}
			}
		}
		out = diff::safe_diff_msg(expects, gots);
	}
	return out;
}

std::string compare_graph (std::istream& expectstr, teq::TensptrT root,
	bool showshape, LabelsMapT labels)
{
	return compare_graph(expectstr, root.get(), showshape, labels);
}

}

#endif
