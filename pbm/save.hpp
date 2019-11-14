///
/// save.hpp
/// pbm
///
/// Purpose:
/// Define functions for marshaling equation graph
///

#include <list>
#include <unordered_set>

#include "teq/traveler.hpp"
#include "teq/ifunctor.hpp"

#include "tag/tag.hpp"

#include "pbm/marshal.hpp"

#ifndef PBM_SAVE_HPP
#define PBM_SAVE_HPP

namespace pbm
{

using LeafMarshF = std::function<std::string(teq::iLeaf*)>;

void save_graph (cortenn::Graph& out, teq::TensptrsT roots,
	tag::TagRegistry& registry, LeafMarshF marshal_leaf);

}

#endif // PBM_SAVE_HPP
