#include "soil/external.hpp"

DataBucket evaluate (Nodeptr& exit_node)
{
	Pool pool;
	return DataBucket(
		exit_node->calculate(pool),
		exit_node->type(),
		exit_node->shape());
}

Nodeptr group (Nodeptr a)
{
	OPCODE opcode = RESHAPE;
	return ShapeTransform::get(a, Shape(std::vector<Shape>{a->shape()}));
}

Nodeptr transpose (Nodeptr a)
{
	OPCODE opcode = TRANSPOSE;
	TransPreOperator trans({0, 1, 1, 2});
	return Functor::get({a}, trans, opcode);
}

Nodeptr transpose (Nodeptr a, std::pair<Group,Group> groups)
{
	OPCODE opcode = TRANSPOSE;
	TransPreOperator trans({
		groups.first[0], groups.first[1],
		groups.second[0], groups.second[1]});
	return Functor::get({a}, trans, opcode);
}

Nodeptr transpose (Nodeptr a, std::pair<uint8_t,uint8_t> dims)
{
	OPCODE opcode = TRANSPOSE;
	uint8_t end0 = dims.first + 1;
	uint8_t end1 = dims.second + 1;
	TransPreOperator trans({dims.first, end0, dims.second, end1});
	return Functor::get({a}, trans, opcode);
}

Nodeptr operator + (Nodeptr a, Nodeptr b)
{
	OPCODE opcode = ADD;
	ElemPreOperator elem;
	return Functor::get({a, b}, elem, opcode);
}

Nodeptr operator * (Nodeptr a, Nodeptr b)
{
	OPCODE opcode = MUL;
	ElemPreOperator elem;
	return Functor::get({a, b}, elem, opcode);
}

Nodeptr matmul (Nodeptr a, Nodeptr b)
{
	OPCODE opcode = MATMUL;
	uint8_t idxa = a->shape().group(0).n_rank();
	uint8_t enda = idxa + a->shape().group(1).n_rank();
	uint8_t idxb = b->shape().group(0).n_rank();
	uint8_t endb = idxb + b->shape().group(1).n_rank();
	MatPreOperator mat({idxa, enda}, {idxb, endb});
	return Functor::get({a, b}, mat, opcode);
}
