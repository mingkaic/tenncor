///
/// string.hpp
/// err
///
/// Purpose:
/// Define string conversions for displaying various types
///

#include <algorithm>
#include <string>
#include <sstream>
#include <tuple>

#ifndef ERR_STRING_HPP
#define ERR_STRING_HPP

namespace err
{

/// Symbol for the start of an array as string
const char arr_begin = '[';

/// Symbol for the end of an array as string
const char arr_end = ']';

/// Symbol for the delimter between elements of an array as string
const char arr_delim = '\\';

/// Stream C-style strings to s
void to_stream (std::ostream& s, const char* str);

/// Stream std::strings to s
void to_stream (std::ostream& s, std::string str);

/// Stream byte-size integers and display as numbers to s
void to_stream (std::ostream& s, int8_t c);

/// Stream byte-size unsigned integers and display as numbers to s
void to_stream (std::ostream& s, uint8_t c);

/// Stream generic value to s
template <typename T>
void to_stream (std::ostream& s, T val)
{
	s << val;
}

/// Stream values between iterators as an array
template <typename Iterator>
void to_stream (std::ostream& s, Iterator begin, Iterator end)
{
	s << arr_begin;
	if (begin != end)
	{
		to_stream(s, *(begin++));
		while (begin != end)
		{
			s << arr_delim;
			to_stream(s, *(begin++));
		}
	}
	s << arr_end;
}

/// Return string representation for common arguments
template <typename T>
std::string to_string (T arg)
{
	std::stringstream ss;
	to_stream(ss, arg);
	return ss.str();
}

/// Return string representation of values between iterators
template <typename Iterator>
std::string to_string (Iterator begin, Iterator end)
{
	std::stringstream ss;
	to_stream(ss, begin, end);
	return ss.str();
}

/// Return std::string with snprintf formatting
template <typename... ARGS>
std::string sprintf (std::string format, ARGS... args)
{
	size_t n = std::snprintf(nullptr, 0, format.c_str(), args...) + 1;
	char buf[n];
	std::snprintf(buf, n, format.c_str(), args...);
	return std::string(buf, buf + n - 1);
}

}

#endif // ERR_STRING_HPP
