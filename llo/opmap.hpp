///
/// opmap.hpp
/// llo
///
/// Purpose:
/// Associate age::OPCODE to operations on data
///

#include "llo/generated/runtime.hpp"

#include "llo/data.hpp"
#include "llo/helper.hpp"

#ifndef LLO_OPMAP_HPP
#define LLO_OPMAP_HPP

namespace llo
{

template <age::_GENERATED_OPCODES OP, typename T>
struct Executer
{
	static void exec (GenericData& out, DataArgsT& data)
	{
		err::fatalf("cannot %s of type %s", age::opname(OP).c_str(),
			nametype(get_type<T>()).c_str());
	}
};

#define UNARY(OP, METHOD)template <typename T>\
struct Executer<age::OP,T>{\
static void exec (GenericData& out, DataArgsT& data) {\
METHOD((T*) out.data_.get(), to_ref<T>(data[0])); } };

UNARY(ABS, abs)
UNARY(NEG, neg)
UNARY(SIN, sin)
UNARY(COS, cos)
UNARY(TAN, tan)
UNARY(EXP, exp)
UNARY(LOG, log)
UNARY(SQRT, sqrt)
UNARY(ROUND, round)

#undef UNARY

#define BINARY(OP, METHOD)template <typename T>\
struct Executer<age::OP,T>{\
static void exec (GenericData& out, DataArgsT& data) {\
METHOD((T*) out.data_.get(), out.shape_, to_ref<T>(data[0]), to_ref<T>(data[1])); } };

BINARY(POW, pow)
BINARY(SUB, sub)
BINARY(DIV, div)
BINARY(EQ, eq)
BINARY(NEQ, neq)
BINARY(LT, lt)
BINARY(GT, gt)

template <typename T>
struct Executer<age::RAND_BINO,T>
{
	static void exec (GenericData& out, DataArgsT& data)
	{
		rand_binom((T*) out.data_.get(), out.shape_,
			VecRef<T>{data[0].first,
				(T*) data[0].second.data_.get(),
				data[0].second.shape_},
			VecRef<double>{data[1].first,
				(double*) data[1].second.data_.get(),
				data[1].second.shape_});
	}
};

BINARY(RAND_UNIF, rand_uniform)
BINARY(RAND_NORM, rand_normal)

#undef BINARY

#define NARY(OP, METHOD)template <typename T>\
struct Executer<age::OP,T> {\
static void exec (GenericData& out, DataArgsT& data){\
METHOD((T*) out.data_.get(), out.shape_, to_refs<T>(data)); } };

NARY(SUM, add)
NARY(PROD, mul)
NARY(MIN, min)
NARY(MAX, max)

#undef NARY

template <age::_GENERATED_OPCODES OP>
void exec (GenericData& out, DataArgsT& data)
{
	switch (out.dtype_)
	{
		case DOUBLE:
			Executer<OP,double>::exec(out, data);
		break;
		case FLOAT:
			Executer<OP,float>::exec(out, data);
		break;
		case INT8:
			Executer<OP,int8_t>::exec(out, data);
		break;
		case INT16:
			Executer<OP,int16_t>::exec(out, data);
		break;
		case INT32:
			Executer<OP,int32_t>::exec(out, data);
		break;
		case INT64:
			Executer<OP,int64_t>::exec(out, data);
		break;
		case UINT8:
			Executer<OP,uint8_t>::exec(out, data);
		break;
		case UINT16:
			Executer<OP,uint16_t>::exec(out, data);
		break;
		case UINT32:
			Executer<OP,uint32_t>::exec(out, data);
		break;
		case UINT64:
			Executer<OP,uint64_t>::exec(out, data);
		break;
		default:
			err::fatal("executing bad type");
	}
}

void op_exec (age::_GENERATED_OPCODES opcode, GenericData& out, DataArgsT& data);

}

#endif // LLO_OPMAP_HPP
