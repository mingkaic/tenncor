#include <functional>

#include "soil/inode.hpp"
#include "soil/functor.hpp"

#ifndef SHAPER_HPP
#define SHAPER_HPP

using Shaper = std::function<Shape(std::vector<iNode*>)>;

Shape elem_shaper (std::vector<iNode*> args);

Shape transpose_shaper (std::vector<iNode*> args);

Shape matmul_shaper (std::vector<iNode*> args);

Shaper get_shaper (OPCODE opcode);

#endif /* SHAPER_HPP */
