#include "dbg/ade_csv.hpp"

void multiline_replace (std::string& multiline)
{
	size_t i = 0;
	char nline = '\n';
	while ((i = multiline.find(nline, i)) != std::string::npos)
	{
		multiline.replace(i, 1, "\\");
	}
}
