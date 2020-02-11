#ifndef EXPERIMENTAL_OPT_APPLY_HPP
#define EXPERIMENTAL_OPT_APPLY_HPP

#include "experimental/opt/rule.hpp"
#include "experimental/opt/graph.hpp"

namespace opt
{

// Returns true if at least one rule is applied.
// performs a single run of conversion rules,
// but does not guarantee complete optimization
// (further optimization may be needed)
bool optimize (GraphInfo& graph, const OptRulesT& rules);

}

#endif // EXPERIMENTAL_OPT_APPLY_HPP
