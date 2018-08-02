#include "soil/external.hpp"

DataBucket evaluate (Nodeptr& exit_node)
{
	Session sess;
	return DataBucket(
		exit_node->calculate(sess),
		exit_node->type(),
		exit_node->shape());
}

CoordOp dim_swap (std::pair<uint8_t,uint8_t> dims)
{
	return [dims](std::vector<DimT>& coords)
	{
		std::swap(coords[dims.first], coords[dims.second]);
	};
}

Nodeptr group (Nodeptr a)
{
	return ShapeTransform::get(a, Shape(std::vector<Shape>{a->shape()}));
}

Nodeptr transpose (Nodeptr a)
{
	return Functor::get({a}, OPCODE::TRANSPOSE);
}

Nodeptr transpose (Nodeptr a, CoordOp swapdim)
{
	return Copyover::get(a, swapdim);
}

Nodeptr operator + (Nodeptr a, Nodeptr b)
{
	return Functor::get({a, b}, OPCODE::ADD);
}

Nodeptr operator * (Nodeptr a, Nodeptr b)
{
	return Functor::get({a, b}, OPCODE::MUL);
}

Nodeptr matmul (Nodeptr a, Nodeptr b)
{
	return Functor::get({a, b}, OPCODE::MATMUL);
}
