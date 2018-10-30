///
///	shear.hpp
///	llo
///
///	Purpose:
///	Define llo graph pruning functions
///

#include "llo/traveler.hpp"

#ifndef LLO_SHEAR_HPP
#define LLO_SHEAR_HPP

namespace llo
{

/// Return tree that prunes zero branches in input according to OPCODE
/// For example, add(x, 0) is converted to simply x, while mul(x, 0) is 0
DataNode zero_prune (DataNode root);

}

#endif // LLO_SHEAR_HPP
