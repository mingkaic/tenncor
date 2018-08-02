#include "sand/operator.hpp"
#include "util/error.hpp"
#include "util/mapper.hpp"

#include "sand/include/matmul.hpp"

#ifdef OPERATOR_HPP

using TypedOperations = EnumMap<DTYPE,Operation>;

#define TYPE_FUNC(FUNC) TypedOperations{\
{ DTYPE::DOUBLE, FUNC<double> },{ DTYPE::FLOAT, FUNC<float> },\
{ DTYPE::INT8, FUNC<int8_t> },{ DTYPE::UINT8, FUNC<uint8_t> },\
{ DTYPE::INT16, FUNC<int16_t> },{ DTYPE::UINT16, FUNC<uint16_t> },\
{ DTYPE::INT32, FUNC<int32_t> },{ DTYPE::UINT32, FUNC<uint32_t> },\
{ DTYPE::INT64, FUNC<int64_t> },{ DTYPE::UINT64, FUNC<uint64_t> } }

static EnumMap<OPCODE,TypedOperations> operations =
{
	{TRANSPOSE, TYPE_FUNC(transpose)},
	{ADD, TYPE_FUNC(add)},
	{MUL, TYPE_FUNC(mul)},
	{MATMUL, TYPE_FUNC(matmul)},
};

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
