#include <functional>

#include "sand/inode.hpp"
#include "sand/opcode.hpp"

#ifndef SOIL_GRADER_HPP
#define SOIL_GRADER_HPP

using Grader = std::function<Nodeptr(std::vector<Nodeptr>,Nodeptr&)>;

Nodeptr add_grad (std::vector<Nodeptr> args, Nodeptr& wrt);

Nodeptr mul_grad (std::vector<Nodeptr> args, Nodeptr& wrt);

Nodeptr matmul_grad (std::vector<Nodeptr> args, Nodeptr& wrt);

Grader get_grader (OPCODE opcode);

#endif /* SOIL_GRADER_HPP */
