#include "util/strify.hpp"

#ifdef UTIL_STRIFY_HPP

namespace util
{

void to_stream (std::ostream&) {}

void to_stream (std::ostream& s, const char* str)
{
	to_stream(s, std::string(str));
}

void to_stream (std::ostream& s, std::string str)
{
	for (size_t i = 0, n = str.size(); i < n; ++i)
	{
		switch (str[i]) {
			case util::BEGIN:
			[[fallthrough]];
			case util::END:
			[[fallthrough]];
			case util::DELIM:
				str.insert(str.begin() + i, util::DELIM);
				++i;
				++n;
		}
	}
	s << str;
}

void to_stream (std::ostream& s, int8_t c)
{
	s << (int) c;
}

void to_stream (std::ostream& s, uint8_t c)
{
	s << (unsigned) c;
}

}

#endif
