#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

#ifndef STDUTIL_MAP_HPP
#define STDUTIL_MAP_HPP

namespace util
{

template <typename SEARCHABLE, typename KEYTYPE>
bool has (const SEARCHABLE& s, const KEYTYPE& key)
{
	return s.end() != s.find(key);
}

}

#endif // STDUTIL_MAP_HPP
