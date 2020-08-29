#include "diff/diff.hpp"

#include "testutil/tutil.hpp"

#ifdef TEST_TUTIL_HPP

namespace tutil
{

const size_t peek_limit = 256;

const std::unordered_set<char> cset = {' ', '\t', '\n', default_indent};

std::string compare_graph (std::istream& expectstr,
	teq::iTensor* root, const PrintEqConfig& printopt)
{
	PrettyEquation artist;
	artist.cfg_ = printopt;
	std::stringstream gotstr;
	artist.print(gotstr, root);
	std::string out;
	bool exrun, gorun;
	exrun = gorun = true;
	while (out.empty() && (exrun || gorun))
	{
		types::StringsT expects;
		types::StringsT gots;
		std::string line;
		for (size_t i = 0; i < peek_limit && exrun; ++i)
		{
			exrun = (bool) std::getline(expectstr, line);
			if (exrun)
			{
				fmts::strip(line, cset);
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
				fmts::strip(line, {'_', ' ', '\t', '\n'});
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

std::string compare_graph (std::istream& expectstr,
	teq::TensptrT root, const PrintEqConfig& printopt)
{
	return compare_graph(expectstr, root.get(), printopt);
}

}

#endif
