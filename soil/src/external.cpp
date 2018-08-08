#include "soil/external.hpp"

DataBucket evaluate (Nodeptr& exit_node)
{
	Pool pool;
	return DataBucket(
		exit_node->calculate(pool),
		exit_node->type(),
		exit_node->shape());
}

Nodeptr cast (Nodeptr a, DTYPE type)
{
	OPCODE opcode = TYPECAST;
	TypecastPreOperator tcast(type, a->type());
	return Functor::get({a}, tcast, opcode);
}

Nodeptr abs (Nodeptr a)
{
	OPCODE opcode = ABS;
	ElemPreOperator elem;
	return Functor::get({a}, elem, opcode);
}

Nodeptr operator - (Nodeptr a)
{
	OPCODE opcode = NEG;
	ElemPreOperator elem;
	return Functor::get({a}, elem, opcode);
}

Nodeptr operator ! (Nodeptr a)
{
	OPCODE opcode = NOT;
	ElemPreOperator elem;
	return Functor::get({a}, elem, opcode);
}

Nodeptr sin (Nodeptr a)
{
	OPCODE opcode = SIN;
	ElemPreOperator elem;
	return Functor::get({a}, elem, opcode);
}

Nodeptr cos (Nodeptr a)
{
	OPCODE opcode = COS;
	ElemPreOperator elem;
	return Functor::get({a}, elem, opcode);
}

Nodeptr tan (Nodeptr a)
{
	OPCODE opcode = TAN;
	ElemPreOperator elem;
	return Functor::get({a}, elem, opcode);
}

Nodeptr exp (Nodeptr a)
{
	OPCODE opcode = EXP;
	ElemPreOperator elem;
	return Functor::get({a}, elem, opcode);
}

Nodeptr log (Nodeptr a)
{
	OPCODE opcode = LOG;
	ElemPreOperator elem;
	return Functor::get({a}, elem, opcode);
}

Nodeptr sqrt (Nodeptr a)
{
	OPCODE opcode = SQRT;
	ElemPreOperator elem;
	return Functor::get({a}, elem, opcode);
}

Nodeptr round (Nodeptr a)
{
	OPCODE opcode = ROUND;
	ElemPreOperator elem;
	return Functor::get({a}, elem, opcode);
}

Nodeptr flip (Nodeptr a, uint dim)
{
	OPCODE opcode = FLIP;
	FlipPreOperator inst(dim);
	return Functor::get({a}, inst, opcode);
}

Nodeptr transpose (Nodeptr a)
{
	OPCODE opcode = TRANSPOSE;
	TransPreOperator trans({0, 1, 1, 2});
	return Functor::get({a}, trans, opcode);
}

Nodeptr transpose (Nodeptr a, std::pair<Range,Range> groups)
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

Nodeptr n_elems (Nodeptr a)
{
	OPCODE opcode = N_ELEMS;
	NElemsPreOperator nelems;
	return Functor::get({a}, nelems, opcode);
}

Nodeptr n_dims (Nodeptr a)
{
	OPCODE opcode = N_DIMS;
	NDimsPreOperator ndims;
	return Functor::get({a}, ndims, opcode);
}

Nodeptr n_dims (Nodeptr a, uint8_t d)
{
	OPCODE opcode = N_DIMS;
	NDimsPreOperator ndims(d);
	return Functor::get({a}, ndims, opcode);
}

Nodeptr group (Nodeptr a)
{
	OPCODE opcode = COPY;
	GroupPreOperator inst({0, a->shape().n_rank()});
	return Functor::get({a}, inst, opcode);
}

Nodeptr group (Nodeptr a, Range range)
{
	OPCODE opcode = COPY;
	GroupPreOperator inst(range);
	return Functor::get({a}, inst, opcode);
}

Nodeptr pow (Nodeptr base, Nodeptr xponent)
{
	OPCODE opcode = POW;
	ElemPreOperator elem;
	return Functor::get({base, xponent}, elem, opcode);
}

Nodeptr operator + (Nodeptr a, Nodeptr b)
{
	OPCODE opcode = ADD;
	ElemPreOperator elem;
	return Functor::get({a, b}, elem, opcode);
}

Nodeptr operator - (Nodeptr a, Nodeptr b)
{
	OPCODE opcode = SUB;
	ElemPreOperator elem;
	return Functor::get({a, b}, elem, opcode);
}

Nodeptr operator * (Nodeptr a, Nodeptr b)
{
	OPCODE opcode = MUL;
	ElemPreOperator elem;
	return Functor::get({a, b}, elem, opcode);
}

Nodeptr operator / (Nodeptr a, Nodeptr b)
{
	OPCODE opcode = DIV;
	ElemPreOperator elem;
	return Functor::get({a, b}, elem, opcode);
}

Nodeptr operator == (Nodeptr a, Nodeptr b)
{
	OPCODE opcode = EQ;
	ElemPreOperator elem;
	return Functor::get({a, b}, elem, opcode);
}

Nodeptr operator != (Nodeptr a, Nodeptr b)
{
	OPCODE opcode = NE;
	ElemPreOperator elem;
	return Functor::get({a, b}, elem, opcode);
}

Nodeptr operator < (Nodeptr a, Nodeptr b)
{
	OPCODE opcode = LT;
	ElemPreOperator elem;
	return Functor::get({a, b}, elem, opcode);
}

Nodeptr operator > (Nodeptr a, Nodeptr b)
{
	OPCODE opcode = GT;
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

Nodeptr binomial_sample (Nodeptr n, Nodeptr p)
{
	OPCODE opcode = BINO;
	BinomPreOperator bino;
	return Functor::get({n, p}, bino, opcode);
}

Nodeptr uniform_sample (Nodeptr a, Nodeptr b)
{
	OPCODE opcode = UNIF;
	ElemPreOperator elem;
	return Functor::get({a, b}, elem, opcode);
}

Nodeptr normal_sample (Nodeptr mean, Nodeptr stdev)
{
	OPCODE opcode = NORM;
	ElemPreOperator elem;
	return Functor::get({mean, stdev}, elem, opcode);
}
Nodeptr arg_max (Nodeptr a)
{
	OPCODE opcode = ARGMAX;
	ReducePreOperator red;
	return Functor::get({a}, red, opcode);
}

Nodeptr arg_max (Nodeptr a, uint64_t dim)
{
	OPCODE opcode = ARGMAX;
	ReducePreOperator red(dim);
	return Functor::get({a}, red, opcode);
}

Nodeptr reduce_max (Nodeptr a)
{
	OPCODE opcode = RMAX;
	ReducePreOperator red;
	return Functor::get({a}, red, opcode);
}

Nodeptr reduce_max (Nodeptr a, uint64_t dim)
{
	OPCODE opcode = RMAX;
	ReducePreOperator red(dim);
	return Functor::get({a}, red, opcode);
}

Nodeptr reduce_sum (Nodeptr a)
{
	OPCODE opcode = RSUM;
	ReducePreOperator red;
	return Functor::get({a}, red, opcode);
}

Nodeptr reduce_sum (Nodeptr a, uint64_t dim)
{
	OPCODE opcode = RSUM;
	ReducePreOperator red(dim);
	return Functor::get({a}, red, opcode);
}
