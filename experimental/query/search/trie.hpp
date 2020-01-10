#ifndef ESTD_TRIE_HPP
#define ESTD_TRIE_HPP

#include <algorithm>
#include <array>
#include <memory>
#include <unordered_map>
#include <stdexcept>
#include <optional>

#include "logs/logs.hpp"

#include "estd/estd.hpp"

namespace estd
{

template <typename KEY, typename VAL, typename HASHER>
struct TrieNode final
{
	// Node always recursively delete children to avoid memory leak,
	// to avoid invalid deletion, set to change of ownership/deleted child to null
	~TrieNode (void)
	{
		for (auto cpair : children_)
		{
			delete cpair.second;
		}
	}

	std::unordered_map<KEY,TrieNode*,HASHER> children_;

	std::optional<VAL> leaf_;
};

template <typename ARR>
using ArrValT = typename std::iterator_traits<
	typename ARR::iterator>::value_type;

template <typename VECKEY, typename VAL,
	typename HASHER=std::hash<ArrValT<VECKEY>>> // todo: replace w/ c++2a concepts
struct Trie
{
	using TrieNodeT = TrieNode<ArrValT<VECKEY>,VAL,HASHER>;

	// assert keys only consists of lower-case characters
	void add (const VECKEY& keys, VAL val)
	{
		auto hit = keys.begin();
		auto het = keys.end();
		TrieNodeT* it = prefix_find(hit, het);
		TrieNodeT* next;
		if (nullptr == it)
		{
			return;
		}
		for (; hit != het; ++hit)
		{
			next = estd::try_get(it->children_, *hit, nullptr);
			if (nullptr == next)
			{
				next = new TrieNodeT();
				it->children_.emplace(*hit, next);
			}
			it = next;
		}
		it->leaf_ = val;
	}

	VAL& emplace (const VECKEY& keys, VAL val)
	{
		auto hit = keys.begin();
		auto het = keys.end();
		TrieNodeT* it = prefix_find(hit, het);
		TrieNodeT* next;
		if (nullptr == it)
		{
			logs::fatal("cannot emplace empty keys");
		}
		for (; hit != het; ++hit)
		{
			next = estd::try_get(it->children_, *hit, nullptr);
			if (nullptr == next)
			{
				next = new TrieNodeT();
				it->children_.emplace(*hit, next);
			}
			it = next;
		}
		if (false == it->leaf_.has_value())
		{
			it->leaf_ = val;
		}
		return *(it->leaf_);
	}

	void rm (const VECKEY& keys)
	{
		if (0 == keys.size())
		{
			return;
		}

		TrieNodeT* it = root_.get();
		TrieNodeT* parent = it;
		const auto* parent_next = &keys[0];
		// traverse to the first node where trie differs from word
		for (const auto& h : keys)
		{
			// move root downward to avoid parents with multiple children
			if (1 < it->children_.size())
			{
				parent = it;
				parent_next = &h;
			}
			it = estd::try_get(it->children_, h, nullptr);
			if (nullptr == it)
			{
				return; // not found
			}
		}
		it->leaf_.reset();
		// assert that path from parent to it has no sibling branches
		if (it->children_.size() == 0)
		{
			delete parent->children_.at(*parent_next);
			parent->children_.erase(*parent_next);
		}
	}

	VAL& at (const VECKEY& keys)
	{
		auto hit = keys.begin();
		auto het = keys.end();
		auto found = prefix_find(hit, het);
		if (nullptr == found && hit != het && false == found->leaf_.has_value())
		{
			logs::fatalf("key %s not found",
				fmts::to_string(keys.begin(), keys.end()).c_str());
		}
		return *found->leaf_;
	}

	const VAL& at (const VECKEY& keys) const
	{
		auto hit = keys.begin();
		auto het = keys.end();
		auto found = prefix_find(hit, het);
		if (nullptr == found && hit != het && false == found->leaf_.has_value())
		{
			logs::fatalf("key %s not found",
				fmts::to_string(keys.begin(), keys.end()).c_str());
		}
		return *found->leaf_;
	}

	bool contains_exact (const VECKEY& keys) const
	{
		auto hit = keys.begin();
		auto het = keys.end();
		auto found = prefix_find(hit, het);
		return nullptr != found && hit == het && found->leaf_.has_value();
	}

	bool contains_prefix (const VECKEY& prefix) const
	{
		auto hit = prefix.begin();
		auto het = prefix.end();
		auto found = prefix_find(hit, het);
		return nullptr != found && hit == het;
	}

	VECKEY best_prefix (const VECKEY& keys) const
	{
		auto hit = keys.begin();
		prefix_find(hit, keys.end());
		return VECKEY(keys.begin(), hit);
	}

private:
	using KeyIt = typename VECKEY::const_iterator;

	TrieNodeT* prefix_find (KeyIt& hash_it, KeyIt hash_et)
	{
		if (hash_it == hash_et)
		{
			return nullptr;
		}
		TrieNodeT* out = root_.get();
		// traverse to the first node where trie differs from word
		for (; hash_it != hash_et &&
			estd::has(out->children_, *hash_it); ++hash_it)
		{
			out = out->children_.at(*hash_it);
		}
		return out;
	}

	const TrieNodeT* prefix_find (KeyIt& hash_it, KeyIt hash_et) const
	{
		if (hash_it == hash_et)
		{
			return nullptr;
		}
		const TrieNodeT* out = root_.get();
		// traverse to the first node where trie differs from word
		for (; hash_it != hash_et &&
			estd::has(out->children_, *hash_it); ++hash_it)
		{
			out = out->children_.at(*hash_it);
		}
		return out;
	}

	std::unique_ptr<TrieNodeT> root_ = std::make_unique<TrieNodeT>();
};

}

#endif // ESTD_TRIE_HPP
