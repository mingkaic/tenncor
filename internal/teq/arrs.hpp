#include <numeric>
#include <vector>

#ifndef TEQ_ARRS_HPP
#define TEQ_ARRS_HPP

namespace arrs
{

template <typename T>
std::vector<T> range (T begin, T end)
{
	std::vector<T> out(end - begin);
	std::iota(out.begin(), out.end(), begin);
	return out;
}

template <typename T>
std::vector<T> concat (std::vector<T> l, const std::vector<T>& r)
{
	l.insert(l.end(), r.begin(), r.end());
	return l;
}

template <typename T, typename ...ARGS>
std::vector<T> concat (std::vector<T> l,
	const std::vector<T>& r, ARGS... other)
{
	return concat(l, concat(r, other...));
}

}

#endif // TEQ_ARRS_HPP
