#include <algorithm>
#include <string>
#include <sstream>
#include <tuple>
#include <vector>

#ifndef UTIL_STRIFY_HPP
#define UTIL_STRIFY_HPP

namespace util
{

const char BEGIN = '[';

const char END = ']';

const char DELIM = '\\';

void to_stream (std::ostream& s);

void to_stream (std::ostream& s, const char* str);

void to_stream (std::ostream& s, std::string str);

void to_stream (std::ostream& s, int8_t c);

void to_stream (std::ostream& s, uint8_t c);

template <typename T>
void to_stream (std::ostream& s, T val)
{
	s << val;
}

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

template <typename T, typename... Args>
void to_stream (std::ostream& s, T val, Args... args)
{
	to_stream(s, val);
	s << DELIM;
	to_stream(s, args...);
}

template <typename Tuple, size_t... I>
std::string tuple_to_string (const Tuple& tp, std::index_sequence<I...>)
{
	std::stringstream ss;
	to_stream(ss, std::get<I>(tp)...);
	return ss.str();
}

template <typename... Args>
std::string tuple_to_string (const std::tuple<Args...>& tp)
{
	return tuple_to_string(tp, std::index_sequence_for<Args...>());
}

}

#endif /* UTIL_STRIFY_HPP */
