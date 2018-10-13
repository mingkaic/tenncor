///
/// string.hpp
/// ade
///
/// Purpose:
/// Define string conversions for displaying various types
///

#include <algorithm>
#include <string>
#include <sstream>
#include <tuple>
#include <vector>

#ifndef ADE_STRING_HPP
#define ADE_STRING_HPP

namespace ade
{

const char BEGIN = '[';

const char END = ']';

const char DELIM = '\\';

/// Do nothing to stream, needed to terminate template
void to_stream (std::ostream& s);

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

/// Stream generic vector to s
template <typename T>
void to_stream (std::ostream& s, std::vector<T> vec)
{
	s << BEGIN;
	if (vec.size() > 0)
	{
		to_stream(s, vec[0]);
		for (size_t i = 1, n = vec.size(); i < n; ++i)
		{
			s << DELIM;
			to_stream(s, vec[i]);
		}
	}
	s << END;
}

/// Stream variadic args to s
template <typename T, typename... Args>
void to_stream (std::ostream& s, T val, Args... args)
{
	to_stream(s, val);
	s << DELIM;
	to_stream(s, args...);
}

/// Return string representation of a tuple content given a specific sequence
template <typename Tuple, size_t... I>
std::string to_string (const Tuple& tp, std::index_sequence<I...>)
{
	std::stringstream ss;
	to_stream(ss, std::get<I>(tp)...);
	return ss.str();
}

/// Return string representation of a tuple content in order stored
template <typename... Args>
std::string to_string (const std::tuple<Args...>& tp)
{
	return to_string(tp, std::index_sequence_for<Args...>());
}

/// Return string representation for common arguments
template <typename T>
std::string to_string (T arg)
{
	std::stringstream ss;
	to_stream(ss, arg);
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

#endif // ADE_STRING_HPP
