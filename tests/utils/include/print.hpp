#include <cstdarg>
#include <sstream>

#ifndef TTEST_PRINT_HPP
#define TTEST_PRINT_HPP

namespace testutils
{

template <typename T>
void print (std::vector<T> raw, std::ostream& os = std::cout)
{
	if (raw.empty())
	{
		os << "empty";
	}
	else
	{
		for (T r : raw)
		{
			os << r << " ";
		}
		os << "\n";
	}
}

std::string sprintf (const char* fmt...);

}

#endif
