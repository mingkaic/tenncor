#ifndef SEARCH_SINDEX_HPP
#define SEARCH_SINDEX_HPP

#include <list>

#include <boost/functional/hash.hpp>

#include "teq/ifunctor.hpp"
#include "teq/traveler.hpp"

#include "eigen/generated/opcode.hpp"

#include "experimental/query/search/trie.hpp"

namespace query
{

struct PathNode final
{
	friend bool operator == (const PathNode& a, const PathNode& b)
	{
		return a.idx_ == b.idx_ && a.op_ == b.op_;
	}

	size_t idx_;

	egen::_GENERATED_OPCODE op_;
};

using PathNodesT = std::vector<PathNode>;

namespace search
{

struct PathNodeHasher final
{
	size_t operator() (const PathNode& node) const
	{
		size_t seed = 0;
		boost::hash_combine(seed, node.op_);
		boost::hash_combine(seed, node.idx_);
		return seed;
	}
};

// path root R to leaf L key maps to mapping L -> []R
using PathVal = teq::LeafMapT<teq::FuncSetT>;

// Rationality:
// # of Path Keys = # Leaves
// size of PathVal value per leaf is proportional to length of max Path
// Realistically, number of edges of an operational graph

using OpTrieT = estd::Trie<PathNodesT,PathVal,PathNodeHasher>;

/// Populate index table of graph structure
void populate_itable (OpTrieT& itable, teq::TensptrsT roots);

}

}

#endif // SEARCH_SINDEX_HPP
