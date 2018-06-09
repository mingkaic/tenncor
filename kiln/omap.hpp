/*!
 *
 *  omap.hpp
 *  kiln
 *
 *  Purpose:
 *  ordered map
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include <list>
#include <unordered_map>
#include <experimental/optional>

#pragma once
#ifndef KILN_OMAP_HPP
#define KILN_OMAP_HPP

namespace kiln
{

using namespace std::experimental;

template <typename T>
using list_it = typename std::list<T>::iterator;

template <typename T>
using list_const_it = typename std::list<T>::const_iterator;

template <typename K, typename V>
class OrderedMap
{
public:
	optional<V> get (K key) const
	{
		optional<V> out;
		auto it = dict_.find(key);
		if (dict_.end() != it)
		{
			out = *(it->second);
		}
		return out;
	}

	bool has (K key) const
	{
		return dict_.end() != dict_.find(key);
	}

	size_t size (void) const
	{
		return order_.size();
	}

	list_it<V> begin (void)
	{
		return order_.begin();
	}

	list_it<V> end (void)
	{
		return order_.end();
	}

	list_const_it<V> begin (void) const
	{
		return order_.cbegin();
	}

	list_const_it<V> end (void) const
	{
		return order_.cend();
	}

	bool put (K key, V value)
	{
		bool success = false == has(key);
		if (success)
		{
			order_.push_back(value);
			dict_[key] = std::prev(order_.end());
		}
		return success;
	}

	bool remove (K key)
	{
		auto it = dict_.find(key);
		bool success = dict_.end() != it;
		if (success)
		{
			auto vt = it->second;
			order_.erase(vt);
			dict_.erase(it);
		}
		return success;
	}

	bool replace (K key, V value)
	{
		bool success = remove(key);
		if (success)
		{
			success = put(key, value);
		}
		return success;
	}

private:
	std::list<V> order_;

	std::unordered_map<K, list_it<V> > dict_;
};

}

#endif /* KILN_OMAP_HPP */
