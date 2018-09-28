/*!
 *
 *  grader.hpp
 *  ade
 *
 *  Purpose:
 *  define gradient functions mapped to opcode
 *
 */

#include "ade/tensor.hpp"
#include "ade/opcode.hpp"

#ifndef ADE_GRADER_HPP
#define ADE_GRADER_HPP

namespace ade
{

template <OPCODE opcode, typename... Args>
Tensorptr grader (std::vector<Tensorptr> args, Tensorptr& wrt, Args... meta)
{
	throw std::bad_function_call();
}

_SIGNATURE_DEF(Tensorptr, grader, std::vector<Tensorptr>, Tensorptr& wrt)

}

#endif /* ADE_GRADER_HPP */
