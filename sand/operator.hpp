#include <functional>

#include "sand/meta.hpp"
#include "sand/opcode.hpp"

#ifndef SAND_OPERATOR_HPP
#define SAND_OPERATOR_HPP

struct NodeInfo
{
	char* data_;
	Shape shape_;
};

using Operation = std::function<void(NodeInfo&,std::vector<NodeInfo>&,MetaEncoder::MetaData)>;

bool has_op (OPCODE opcode, DTYPE type);

Operation get_op (OPCODE opcode, DTYPE type);

#endif /* SAND_OPERATOR_HPP */
