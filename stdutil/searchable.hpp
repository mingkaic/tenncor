#include <algorithm>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "logs/logs.hpp"

#ifndef STDUTIL_MAP_HPP
#define STDUTIL_MAP_HPP

namespace util
{

template <typename MAPPABLE>
using ValT = typename MAPPABLE::mapped_type;

template <typename SEARCHABLE, typename KEYTYPE>
bool has (const SEARCHABLE& s, const KEYTYPE& key)
{
	return s.end() != s.find(key);
}

template <typename MAPPABLE, typename KEYTYPE>
bool get (ValT<MAPPABLE>& val, const MAPPABLE& s, const KEYTYPE& key)
{
	auto it = s.find(key);
	bool found = s.end() != it;
	if (found)
	{
		val = it->second;
	}
	return found;
}

template <typename MAPPABLE, typename KEYTYPE>
ValT<MAPPABLE> try_get (const MAPPABLE& s, const KEYTYPE& key,
	ValT<MAPPABLE> default_val)
{
	auto it = s.find(key);
	if (s.end() != it)
	{
		return it->second;
	}
	return default_val;
}

template <typename MAPPABLE, typename KEYTYPE, typename... ARGS>
const ValT<MAPPABLE>& must_getf (
	const MAPPABLE& s, const KEYTYPE& key,
	std::string msg, ARGS... args)
{
	auto it = s.find(key);
	if (s.end() == it)
	{
		logs::fatalf(msg, args...);
	}
	return it->second;
}

template <typename ARR, typename CONTENT>
bool arr_has (const ARR& s, const CONTENT& key)
{
	auto et = s.end();
	return et != std::find(s.begin(), et, key);
}

}

#endif // STDUTIL_MAP_HPP
