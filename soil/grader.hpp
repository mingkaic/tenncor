#include <functional>

#include "sand/inode.hpp"
#include "sand/opcode.hpp"

#ifndef SOIL_GRADER_HPP
#define SOIL_GRADER_HPP

using Grader = std::function<Nodeptr(std::vector<Nodeptr>,Nodeptr&)>;

Nodeptr abs_grad (std::vector<Nodeptr> args, Nodeptr& wrt);

Nodeptr neg_grad (std::vector<Nodeptr> args, Nodeptr& wrt);

Nodeptr logic_not_grad (std::vector<Nodeptr> args, Nodeptr& wrt);

Nodeptr sin_grad (std::vector<Nodeptr> args, Nodeptr& wrt);

Nodeptr cos_grad (std::vector<Nodeptr> args, Nodeptr& wrt);

Nodeptr tan_grad (std::vector<Nodeptr> args, Nodeptr& wrt);

Nodeptr exp_grad (std::vector<Nodeptr> args, Nodeptr& wrt);

Nodeptr log_grad (std::vector<Nodeptr> args, Nodeptr& wrt);

Nodeptr sqrt_grad (std::vector<Nodeptr> args, Nodeptr& wrt);

Nodeptr round_grad (std::vector<Nodeptr> args, Nodeptr& wrt);

Nodeptr flip_grad (std::vector<Nodeptr> args, Nodeptr& wrt);

Nodeptr transpose_grad (std::vector<Nodeptr> args, Nodeptr& wrt);

Nodeptr add_grad (std::vector<Nodeptr> args, Nodeptr& wrt);

Nodeptr mul_grad (std::vector<Nodeptr> args, Nodeptr& wrt);

Nodeptr eq_grad (std::vector<Nodeptr> args, Nodeptr& wrt);

Nodeptr neq_grad (std::vector<Nodeptr> args, Nodeptr& wrt);

Nodeptr lt_grad (std::vector<Nodeptr> args, Nodeptr& wrt);

Nodeptr gt_grad (std::vector<Nodeptr> args, Nodeptr& wrt);

Nodeptr matmul_grad (std::vector<Nodeptr> args, Nodeptr& wrt);

Grader get_grader (OPCODE opcode);

#endif /* SOIL_GRADER_HPP */
