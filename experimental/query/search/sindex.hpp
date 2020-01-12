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
		bool codeq = a.op_ == b.op_;
		if (egen::is_commutative(a.op_) && codeq)
		{
			// ignore idx
			return true;
		}
		return a.idx_ == b.idx_ && codeq;
	}

	friend bool operator < (const PathNode& a, const PathNode& b)
	{
		if (egen::name_op(a.op_) < egen::name_op(b.op_))
		{
			return true;
		}
		return a.idx_ < b.idx_;
	}

	size_t idx_;

	egen::_GENERATED_OPCODE op_;
};

using PathNodesT = std::vector<PathNode>;

namespace search
{

using PathListT = std::list<PathNode>;

struct PathNodeHasher final
{
	size_t operator() (const PathNode& node) const
	{
		size_t seed = egen::is_commutative(node.op_) ? 0 : node.idx_;
		boost::hash_combine(seed, node.op_);
		return seed;
	}
};

// path root R to leaf L key maps to mapping L -> []R
struct PathVal final
{
	teq::LeafMapT<teq::FuncSetT> leaves_;

	teq::FuncSetT attrs_;
};

// Rationality:
// # of Path Keys = # Leaves
// size of PathVal value per leaf is proportional to length of max Path
// Realistically, number of edges of an operational graph

using OpTrieT = estd::Trie<PathNodesT,PathVal,PathNodeHasher>;

/// Populate index table of graph structure
void populate_itable (OpTrieT& itable, teq::TensptrsT roots);

using PathCbF = std::function<void(const PathListT&,const PathVal&)>;

void possible_paths (const PathCbF& cb,
	const OpTrieT& itable, const PathNodesT& path);

}

}

#endif // SEARCH_SINDEX_HPP
