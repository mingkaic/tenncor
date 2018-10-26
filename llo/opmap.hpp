///
/// opmap.hpp
/// llo
///
/// Purpose:
/// Associate ade::OPCODE to operations on data
///

#include "ade/opcode.hpp"

#include "llo/data.hpp"
#include "llo/operator.hpp"

#ifndef LLO_OPMAP_HPP
#define LLO_OPMAP_HPP

namespace llo
{

template <ade::OPCODE OP, typename T>
struct Executer
{
	static void exec (GenericData& out,
		std::vector<GenericData>& data, ARGS... args)
	{
		ade::fatalf("cannot %s of type %s", ade::opname(OP).c_str(),
			nametype(get_type<T>()).c_str());
	}
};

#define UNARY_ELEM(OP, METHOD)template <typename T>\
struct Executer<ade::OP,T> { static void exec\
(GenericData& out, std::vector<GenericData>& data)\
{ METHOD((T*) out.data_.get(), VecRef<T>{(T*) data[0].data_.get(),\
out.shape_.n_elems()}); } };

#define BINARY_ELEM(OP, METHOD)template <typename T>\
struct Executer<ade::OP,T> { static void exec\
(GenericData& out, std::vector<GenericData>& data)\
{ METHOD((T*) out.data_.get(),\
VecRef<T>{(T*) data[0].data_.get(), data[0].shape_.n_elems()},\
VecRef<T>{(T*) data[1].data_.get(), data[1].shape_.n_elems()}); } };

#define NARY_ELEM(OP, METHOD)template <typename T>\
struct Executer<ade::OP,T> { static void exec\
(GenericData& out, std::vector<GenericData>& data)\
{ std::vector<VecRef<T>> args(data.size()); \
std::transform(data.begin(), data.end(), args.begin(), \
[](GenericData& gd) { return VecRef<T>{\
(T*) gd.data_.get(), gd.shape_.n_elems()}; });\
METHOD((T*) out.data_.get(), args); } };

template <typename T>
struct Executer<ade::COPY,T>
{
	static void exec (GenericData& out, std::vector<GenericData>& data)
	{
		copy((T*) out.data_.get(),
			VecRef<T>{(T*) data[0].data_.get(), data[0].shape_.n_elems()},
			VecRef<double>{(double*) data[1].data_.get(),
				data[1].shape_.n_elems()});
	}
};

UNARY_ELEM(ABS, abs)
UNARY_ELEM(NEG, neg)
UNARY_ELEM(NOT, bit_not)
UNARY_ELEM(SIN, sin)
UNARY_ELEM(COS, cos)
UNARY_ELEM(TAN, tan)
UNARY_ELEM(EXP, exp)
UNARY_ELEM(LOG, log)
UNARY_ELEM(SQRT, sqrt)
UNARY_ELEM(ROUND, round)

BINARY_ELEM(POW, pow)
NARY_ELEM(ADD, add)
BINARY_ELEM(SUB, sub)
NARY_ELEM(MUL, mul)
BINARY_ELEM(DIV, div)
BINARY_ELEM(EQ, eq)
BINARY_ELEM(NE, neq)
BINARY_ELEM(LT, lt)
BINARY_ELEM(GT, gt)
NARY_ELEM(MIN, min)
NARY_ELEM(MAX, max)

template <typename T>
struct Executer<ade::RAND_BINO,T>
{
	static void exec (GenericData& out, std::vector<GenericData>& data)
	{
		rand_binom((T*) out.data_.get(),
			VecRef<T>{(T*) data[0].data_.get(), data[0].shape_.n_elems()},
			VecRef<double>{(double*) data[1].data_.get(),
				data[1].shape_.n_elems()});
	}
};

BINARY_ELEM(RAND_UNIF, rand_uniform)
BINARY_ELEM(RAND_NORM, rand_normal)

#undef UNARY_ELEM
#undef BINARY_ELEM
#undef UNARY_REDUCE
#undef UNARY_COPY

template <ade::OPCODE OP>
void exec (GenericData& out,
	std::vector<GenericData>& data)
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
			ade::fatal("executing bad type");
	}
}

void op_exec (ade::OPCODE opcode, GenericData& out,
	std::vector<GenericData>& data)
{
	switch (opcode)
	{
		case ade::COPY:
			exec<ade::COPY>(out, data);
		break;
		case ade::ABS:
			exec<ade::ABS>(out, data);
		break;
		case ade::NEG:
			exec<ade::NEG>(out, data);
		break;
		case ade::NOT:
			exec<ade::NOT>(out, data);
		break;
		case ade::SIN:
			exec<ade::SIN>(out, data);
		break;
		case ade::COS:
			exec<ade::COS>(out, data);
		break;
		case ade::TAN:
			exec<ade::TAN>(out, data);
		break;
		case ade::EXP:
			exec<ade::EXP>(out, data);
		break;
		case ade::LOG:
			exec<ade::LOG>(out, data);
		break;
		case ade::SQRT:
			exec<ade::SQRT>(out, data);
		break;
		case ade::ROUND:
			exec<ade::ROUND>(out, data);
		break;
		case ade::POW:
			exec<ade::POW>(out, data);
		break;
		case ade::ADD:
			exec<ade::ADD>(out, data);
		break;
		case ade::SUB:
			exec<ade::SUB>(out, data);
		break;
		case ade::MUL:
			exec<ade::MUL>(out, data);
		break;
		case ade::DIV:
			exec<ade::DIV>(out, data);
		break;
		case ade::EQ:
			exec<ade::EQ>(out, data);
		break;
		case ade::NE:
			exec<ade::NE>(out, data);
		break;
		case ade::LT:
			exec<ade::LT>(out, data);
		break;
		case ade::GT:
			exec<ade::GT>(out, data);
		break;
		case ade::RAND_BINO:
			exec<ade::RAND_BINO>(out, data);
		break;
		case ade::RAND_UNIF:
			exec<ade::RAND_UNIF>(out, data);
		break;
		case ade::RAND_NORM:
			exec<ade::RAND_NORM>(out, data);
		break;
		default:
			ade::fatal("unknown opcode");
	}
}

}

#endif // LLO_OPMAP_HPP
