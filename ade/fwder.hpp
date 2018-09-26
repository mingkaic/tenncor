/*!
 *
 *  fwder.hpp
 *  ade
 *
 *  Purpose:
 *  define shape transformation functions mapped to opcodes
 *
 */

#include "ade/tensor.hpp"
#include "ade/opcode.hpp"

#ifndef ADE_FWDER_HPP
#define ADE_FWDER_HPP

namespace ade
{

template <OPCODE opcode, typename... Args>
Shape forwarder (std::vector<Tensorptr> tens, Args... args)
{
	throw std::bad_function_call();
}

_SIGNATURE_DEF(Shape, forwarder, std::vector<Tensorptr>)

}

#endif /* ADE_FWDER_HPP */
