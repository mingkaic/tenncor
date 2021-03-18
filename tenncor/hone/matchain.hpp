#ifndef HONE_MATCHAIN_HPP
#define HONE_MATCHAIN_HPP

#include "internal/opt/opt.hpp"
#include "tenncor/eteq/eteq.hpp"

namespace types
{

template <typename K, typename V>
using PairsT = std::vector<std::pair<K,V>>;

}

namespace hone
{

void flatten_matmul_hierarchy (
	types::PairsT<teq::iTensor*,teq::TensptrsT>& chain_roots,
	const opt::GraphInfo& graph);

void matrix_chain (opt::GraphInfo& graph);

}

#endif // HONE_MATCHAIN_HPP

