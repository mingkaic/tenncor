#include "err/string.hpp"

#ifdef ERR_STRING_HPP

namespace err
{

void to_stream (std::ostream& s, const char* str)
{
	to_stream(s, std::string(str));
}

void to_stream (std::ostream& s, std::string str)
{
	for (size_t i = 0, n = str.size(); i < n; ++i)
	{
		switch (str[i]) {
			case arr_begin:
			case arr_end:
			case arr_delim:
				str.insert(str.begin() + i, arr_delim);
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
