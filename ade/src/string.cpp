#include "ade/string.hpp"

#ifdef ADE_STRING_HPP

namespace ade
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
			case ade::BEGIN:
			case ade::END:
			case ade::DELIM:
				str.insert(str.begin() + i, ade::DELIM);
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
