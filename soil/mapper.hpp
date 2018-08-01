#include <unordered_map>

#ifndef MAPPER_HPP
#define MAPPER_HPP

struct EnumHash
{
	template <typename T>
	size_t operator() (T e) const
	{
		return static_cast<size_t>(e);
	}
};

template <typename K, typename V>
using EnumMap = std::unordered_map<K,V,EnumHash>;

#endif /* MAPPER_HPP */
