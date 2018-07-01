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
	optional<V> get (K key) const;

	bool has (K key) const;

	size_t size (void) const;

	list_it<V> begin (void);

	list_it<V> end (void);

	list_const_it<V> begin (void) const;

	list_const_it<V> end (void) const;

	bool put (K key, V value);

	bool remove (K key);

	bool replace (K key, V value);

private:
	std::list<V> order_;

	std::unordered_map<K, list_it<V> > dict_;
};

}

#endif /* KILN_OMAP_HPP */

#include "kiln/include/omap.ipp"
