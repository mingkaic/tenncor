#include "sand/operator.hpp"
#include "util/error.hpp"
#include "util/mapper.hpp"

#include "sand/include/unary.hpp"
#include "sand/include/binary.hpp"
#include "sand/include/matmul.hpp"

#ifdef SAND_OPERATOR_HPP

using namespace sand;

using TypedOperations = EnumMap<DTYPE,Operation>;

#define TYPE_FUNC(FUNC) TypedOperations{\
{ DTYPE::DOUBLE, FUNC<double> },{ DTYPE::FLOAT, FUNC<float> },\
{ DTYPE::INT8, FUNC<int8_t> },{ DTYPE::UINT8, FUNC<uint8_t> },\
{ DTYPE::INT16, FUNC<int16_t> },{ DTYPE::UINT16, FUNC<uint16_t> },\
{ DTYPE::INT32, FUNC<int32_t> },{ DTYPE::UINT32, FUNC<uint32_t> },\
{ DTYPE::INT64, FUNC<int64_t> },{ DTYPE::UINT64, FUNC<uint64_t> } }

static EnumMap<OPCODE,TypedOperations> operations =
{
	{TYPECAST, TYPE_FUNC(typecast)},
	{ABS, TYPE_FUNC(abs)},
	{NEG, TYPE_FUNC(neg)},
	{NOT, TYPE_FUNC(logic_not)},
	{SIN, TYPE_FUNC(sand::sin)},
	{COS, TYPE_FUNC(sand::cos)},
	{TAN, TYPE_FUNC(sand::tan)},
	{EXP, TYPE_FUNC(sand::exp)},
	{LOG, TYPE_FUNC(sand::log)},
	{SQRT, TYPE_FUNC(sand::sqrt)},
	{ROUND, TYPE_FUNC(sand::round)},
	{FLIP, TYPE_FUNC(flip)},
	{TRANSPOSE, TYPE_FUNC(transpose)},
	{N_ELEMS, TYPE_FUNC(n_elems)},
	{N_DIMS, TYPE_FUNC(n_dims)},

	{ARGMAX, TYPE_FUNC(arg_max)},
	{RMAX, TYPE_FUNC(reduce_max)},
	{RSUM, TYPE_FUNC(reduce_sum)},

	{POW, TYPE_FUNC(pow)},
	{ADD, TYPE_FUNC(add)},
	{SUB, TYPE_FUNC(sub)},
	{MUL, TYPE_FUNC(mul)},
	{DIV, TYPE_FUNC(div)},
	{EQ, TYPE_FUNC(eq)},
	{NE, TYPE_FUNC(neq)},
	{LT, TYPE_FUNC(lt)},
	{GT, TYPE_FUNC(gt)},
	{MATMUL, TYPE_FUNC(matmul)},

	{BINO, TYPE_FUNC(rand_binom)},
	{UNIF, TYPE_FUNC(rand_uniform)},
	{NORM, TYPE_FUNC(rand_normal)},
};

bool has_op (OPCODE opcode, DTYPE type)
{
	auto it = operations.find(opcode);
	if (operations.end() == it)
	{
		return false;
	}
	auto typemap = it->second;
	auto tit = typemap.find(type);
	return typemap.end() != tit;
}

Operation get_op (OPCODE opcode, DTYPE type)
{
	auto it = operations.find(opcode);
	if (operations.end() == it)
	{
		handle_error("failed to retrieve operation",
			ErrArg<std::string>("opcode", opname(opcode)));
	}
	auto typemap = it->second;
	auto tit = typemap.find(type);
	if (typemap.end() == tit)
	{
		handle_error("failed to retrieve operation of type",
			ErrArg<std::string>("opcode", opname(opcode)),
			ErrArg<std::string>("type", name_type(type)));
	}
	return tit->second;
}

#endif
