///
/// load.hpp
/// pbm
///
/// Purpose:
/// Define functions for marshal and unmarshal equation graph
///

#include "tag/tag.hpp"

#include "pbm/marshal.hpp"

#ifndef PBM_LOAD_HPP
#define PBM_LOAD_HPP

namespace pbm
{

using LeafUnmarshF = std::function<teq::TensptrT(
	const tenncor::Source&,std::string)>;

using EdgesT = std::vector<std::pair<teq::TensptrT,marsh::Maps>>;

using FuncUnmarshF = std::function<teq::TensptrT(
	std::string,const EdgesT&,marsh::Maps&&)>;

/// Return graph info through out available from in graph
void load_graph (teq::TensptrSetT& roots,
	const tenncor::Graph& pb_graph, tag::TagRegistry& registry,
	LeafUnmarshF unmarshal_leaf, FuncUnmarshF unmarshal_func);

}

#endif // PBM_GRAPH_HPP
