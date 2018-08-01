#include <functional>

#include "soil/inode.hpp"
#include "soil/functor.hpp"

#ifndef GRADER_HPP
#define GRADER_HPP

using Grader = std::function<Nodeptr(std::vector<Nodeptr>,Nodeptr&)>;

Nodeptr add_grad (std::vector<Nodeptr> args, Nodeptr& wrt);

Nodeptr mul_grad (std::vector<Nodeptr> args, Nodeptr& wrt);

Nodeptr matmul_grad (std::vector<Nodeptr> args, Nodeptr& wrt);

Grader get_grader (OPCODE opcode);

#endif /* GRADER_HPP */
