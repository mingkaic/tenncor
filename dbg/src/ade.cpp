#include "diff/msg.hpp"

#include "dbg/ade.hpp"

#ifdef DBG_ADE_HPP

static inline void ltrim(std::string &s)
{
	s.erase(s.begin(), std::find_if(s.begin(), s.end(),
		std::not1(std::ptr_fun<int,int>(std::isspace))));
}

static inline void rtrim(std::string &s)
{
	s.erase(std::find_if(s.rbegin(), s.rend(),
		std::not1(std::ptr_fun<int,int>(std::isspace))).base(), s.end());
}

static inline void trim(std::string &s)
{
	ltrim(s);
	rtrim(s);
}

std::string compare_graph (std::istream& expectstr, ade::TensptrT root,
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
		trim(line);
		if (line.size() > 0)
		{
			expects.push_back(line);
		}
	}
	while (std::getline(gotstr, line))
	{
		trim(line);
		if (line.size() > 0)
		{
			gots.push_back(line);
		}
	}
	return diff::diff_msg(expects, gots);
}

#endif
