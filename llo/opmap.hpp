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

template <ade::OPCODE OP, typename T, typename... ARGS>
struct Executer
{
	static void exec (GenericData& out,
		std::vector<GenericData>& data, ARGS... args)
	{
		std::stringstream ss;
		ade::to_stream(ss, args...);
		ade::fatalf(
			"cannot %s of type %s with args %s", ade::opname(OP).c_str(),
			nametype(get_type<T>()).c_str(), ss.str().c_str());
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

#define UNARY_REDUCE(OP, METHOD)template <typename T>\
struct Executer<ade::OP,T,uint8_t> {\
static void exec (GenericData& out, std::vector<GenericData>& data, uint8_t)\
{ METHOD((T*) out.data_.get(), out.shape_.n_elems(),\
VecRef<T>{(T*) data[0].data_.get(), data[0].shape_.n_elems()}); } };

#define UNARY_COPY(OP)template <typename T>\
struct Executer<ade::OP,T,std::vector<ade::DimT>> {\
static void exec (GenericData& out,\
std::vector<GenericData>& data, std::vector<ade::DimT>)\
{ copyover((T*) out.data_.get(), out.shape_.n_elems(),\
VecRef<T>{(T*) data[0].data_.get(), data[0].shape_.n_elems()}); } };

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

template <typename T>
struct Executer<ade::FLIP,T,uint8_t>
{
	static void exec (GenericData& out,
		std::vector<GenericData>& data, uint8_t dim)
	{
		flip((T*) out.data_.get(), (T*) data[0].data_.get(), out.shape_, dim);
	}
};

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

template <typename T>
struct Executer<ade::N_ELEMS,T>
{
	static void exec (GenericData& out, std::vector<GenericData>& data)
	{
		T* ptr = (T*) out.data_.get();
		n_elems(*ptr, data[0].shape_);
	}
};

template <typename T>
struct Executer<ade::N_DIMS,T,uint8_t>
{
	static void exec (GenericData& out, std::vector<GenericData>& data, uint8_t dim)
	{
		T* ptr = (T*) out.data_.get();
		n_dims(*ptr, data[0].shape_, dim);
	}
};

UNARY_REDUCE(ARGMAX, arg_max)
UNARY_REDUCE(RMAX, reduce_max)
UNARY_REDUCE(RSUM, reduce_sum)

template <typename T>
struct Executer<ade::MATMUL,T>
{
	static void exec (GenericData& out, std::vector<GenericData>& data)
	{
		matmul((T*) out.data_.get(),
			(T*) data[0].data_.get(), (T*) data[1].data_.get(),
			data[0].shape_, data[1].shape_, 1, 1);
	}
};

template <typename T>
struct Executer<ade::MATMUL,T,uint8_t,uint8_t>
{
	static void exec (GenericData& out, std::vector<GenericData>& data,
		uint8_t agroup_idx, uint8_t bgroup_idx)
	{
		matmul((T*) out.data_.get(),
			(T*) data[0].data_.get(), (T*) data[1].data_.get(),
			data[0].shape_, data[1].shape_, agroup_idx, bgroup_idx);
	}
};

template <typename T>
struct Executer<ade::PERMUTE,T,std::vector<uint8_t>>
{
	static void exec (GenericData& out, std::vector<GenericData>& data,
		std::vector<uint8_t> order)
	{
		permute((T*) out.data_.get(),
			(T*) data[0].data_.get(), out.shape_, data[0].shape_, order);
	}
};

UNARY_COPY(EXTEND)

#undef UNARY_ELEM
#undef BINARY_ELEM
#undef UNARY_REDUCE
#undef UNARY_COPY

template <ade::OPCODE OP, typename... ARGS>
void exec (GenericData& out,
	std::vector<GenericData>& data, ARGS... args)
{
	switch (out.dtype_)
	{
		case DOUBLE:
			Executer<OP,double,ARGS...>::exec(out, data, args...);
		break;
		case FLOAT:
			Executer<OP,float,ARGS...>::exec(out, data, args...);
		break;
		case INT8:
			Executer<OP,int8_t,ARGS...>::exec(out, data, args...);
		break;
		case INT16:
			Executer<OP,int16_t,ARGS...>::exec(out, data, args...);
		break;
		case INT32:
			Executer<OP,int32_t,ARGS...>::exec(out, data, args...);
		break;
		case INT64:
			Executer<OP,int64_t,ARGS...>::exec(out, data, args...);
		break;
		case UINT8:
			Executer<OP,uint8_t,ARGS...>::exec(out, data, args...);
		break;
		case UINT16:
			Executer<OP,uint16_t,ARGS...>::exec(out, data, args...);
		break;
		case UINT32:
			Executer<OP,uint32_t,ARGS...>::exec(out, data, args...);
		break;
		case UINT64:
			Executer<OP,uint64_t,ARGS...>::exec(out, data, args...);
		break;
		default:
			ade::fatal("executing bad type");
	}
}

template <typename... ARGS>
void op_exec (ade::OPCODE opcode, GenericData& out,
	std::vector<GenericData>& data, ARGS... args)
{
	switch (opcode)
	{
		case ade::ABS:
			exec<ade::ABS,ARGS...>(out, data, args...);
		break;
		case ade::NEG:
			exec<ade::NEG,ARGS...>(out, data, args...);
		break;
		case ade::NOT:
			exec<ade::NOT,ARGS...>(out, data, args...);
		break;
		case ade::SIN:
			exec<ade::SIN,ARGS...>(out, data, args...);
		break;
		case ade::COS:
			exec<ade::COS,ARGS...>(out, data, args...);
		break;
		case ade::TAN:
			exec<ade::TAN,ARGS...>(out, data, args...);
		break;
		case ade::EXP:
			exec<ade::EXP,ARGS...>(out, data, args...);
		break;
		case ade::LOG:
			exec<ade::LOG,ARGS...>(out, data, args...);
		break;
		case ade::SQRT:
			exec<ade::SQRT,ARGS...>(out, data, args...);
		break;
		case ade::ROUND:
			exec<ade::ROUND,ARGS...>(out, data, args...);
		break;
		case ade::FLIP:
			exec<ade::FLIP,ARGS...>(out, data, args...);
		break;
		case ade::POW:
			exec<ade::POW,ARGS...>(out, data, args...);
		break;
		case ade::ADD:
			exec<ade::ADD,ARGS...>(out, data, args...);
		break;
		case ade::SUB:
			exec<ade::SUB,ARGS...>(out, data, args...);
		break;
		case ade::MUL:
			exec<ade::MUL,ARGS...>(out, data, args...);
		break;
		case ade::DIV:
			exec<ade::DIV,ARGS...>(out, data, args...);
		break;
		case ade::EQ:
			exec<ade::EQ,ARGS...>(out, data, args...);
		break;
		case ade::NE:
			exec<ade::NE,ARGS...>(out, data, args...);
		break;
		case ade::LT:
			exec<ade::LT,ARGS...>(out, data, args...);
		break;
		case ade::GT:
			exec<ade::GT,ARGS...>(out, data, args...);
		break;
		case ade::RAND_BINO:
			exec<ade::RAND_BINO,ARGS...>(out, data, args...);
		break;
		case ade::RAND_UNIF:
			exec<ade::RAND_UNIF,ARGS...>(out, data, args...);
		break;
		case ade::RAND_NORM:
			exec<ade::RAND_NORM,ARGS...>(out, data, args...);
		break;
		case ade::N_ELEMS:
			exec<ade::N_ELEMS,ARGS...>(out, data, args...);
		break;
		case ade::N_DIMS:
			exec<ade::N_DIMS,ARGS...>(out, data, args...);
		break;
		case ade::ARGMAX:
			exec<ade::ARGMAX,ARGS...>(out, data, args...);
		break;
		case ade::RMAX:
			exec<ade::RMAX,ARGS...>(out, data, args...);
		break;
		case ade::RSUM:
			exec<ade::RSUM,ARGS...>(out, data, args...);
		break;
		case ade::MATMUL:
			exec<ade::MATMUL,ARGS...>(out, data, args...);
		break;
		case ade::PERMUTE:
			exec<ade::PERMUTE,ARGS...>(out, data, args...);
		break;
		case ade::EXTEND:
			exec<ade::EXTEND,ARGS...>(out, data, args...);
		break;
		default:
			ade::fatal("unknown opcode");
	}
}

}

#endif // LLO_OPMAP_HPP
