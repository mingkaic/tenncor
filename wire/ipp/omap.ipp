//
//  omap.ipp
//  clay
//

namespace clay
{

template <typename K, typename V>
optional<V> OrderedMap<K, V>::get (K key) const
{
    optional<V> out;
    auto it = dict_.find(key);
    if (dict_.end() != it)
    {
        out = *(it->second);
    }
    return out;
}

template <typename K, typename V>
bool OrderedMap<K, V>::has (K key) const
{
    return dict_.end() != dict_.find(key);
}

template <typename K, typename V>
size_t OrderedMap<K, V>::size (void) const
{
    return order_.size();
}

template <typename K, typename V>
list_it<V> OrderedMap<K, V>::begin (void)
{
    return order_.begin();
}

template <typename K, typename V>
list_it<V> OrderedMap<K, V>::end (void)
{
    return order_.end();
}

template <typename K, typename V>
bool OrderedMap<K, V>::put (K key, V value)
{
    bool success = false == has(key);
    if (success)
    {
        order_.push_back(value);
        dict_[key] = std::prev(order_.end());
    }
    return success;
}

template <typename K, typename V>
bool OrderedMap<K, V>::remove (K key)
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

template <typename K, typename V>
bool OrderedMap<K, V>::replace (K key, V value)
{
    bool success = remove(key);
    if (success)
    {
        success = put(key, value);
    }
    return success;
}

}
