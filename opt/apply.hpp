#ifndef OPT_APPLY_HPP
#define OPT_APPLY_HPP

#include "opt/rule.hpp"
#include "opt/graph.hpp"

namespace opt
{

// Returns true if at least one rule is applied.
// performs a single run of conversion rules,
// but does not guarantee complete optimization
// (further optimization may be needed)
bool optimize (GraphInfo& graph, const OptRulesT& rules);

}

#endif // OPT_APPLY_HPP
