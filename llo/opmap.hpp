#include "ade/opcode.hpp"

#include "llo/data.hpp"
#include "llo/operator.hpp"

template <ade::OPCODE opcode, typename T, typename... Args>
struct Executer
{
	static void exec (GenericData& out,
		std::vector<GenericData>& data, Args... args)
	{
		throw std::bad_function_call();
	}
};

#define UNARY_ELEM(OP, METHOD)template <typename T>\
struct Executer<ade::OP,T> { static void exec\
(GenericData& out, std::vector<GenericData>& data)\
{ METHOD((T*) out.data_.get(),\
(T*) data[0].data_.get(), out.shape_.n_elems()); } };

#define BINARY_ELEM(OP, METHOD)template <typename T>\
struct Executer<ade::OP,T> { static void exec\
(GenericData& out, std::vector<GenericData>& data)\
{ METHOD((T*) out.data_.get(),\
(T*) data[0].data_.get(), data[0].shape_.n_elems(),\
(T*) data[1].data_.get(), data[1].shape_.n_elems()); } };

#define UNARY_REDUCE(OP, METHOD)template <typename T>\
struct Executer<ade::OP,T> {\
static void exec (GenericData& out, std::vector<GenericData>& data)\
{ T* ptr = (T*) out.data_.get();\
METHOD(*ptr, (T*) data[0].data_.get(), data[0].shape_.n_elems()); } };

#define UNARY_COPY(OP)template <typename T>\
struct Executer<ade::OP,T> {\
static void exec (GenericData& out, std::vector<GenericData>& data)\
{ copyover((T*) out.data_.get(), out.shape_.n_elems(),\
(T*) data[0].data_.get(), data[0].shape_.n_elems()); } };

UNARY_ELEM(ABS, abs)
UNARY_ELEM(NEG, neg)
UNARY_ELEM(NOT, logic_not)
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
BINARY_ELEM(ADD, add)
BINARY_ELEM(SUB, sub)
BINARY_ELEM(MUL, mul)
BINARY_ELEM(DIV, div)
BINARY_ELEM(EQ, eq)
BINARY_ELEM(NE, neq)
BINARY_ELEM(LT, lt)
BINARY_ELEM(GT, gt)

template <typename T>
struct Executer<ade::BINO,T>
{
	static void exec (GenericData& out, std::vector<GenericData>& data)
	{
		rand_binom((T*) out.data_.get(),
			(T*) data[0].data_.get(), data[0].shape_.n_elems(),
			(double*) data[1].data_.get(), data[1].shape_.n_elems());
	}
};

BINARY_ELEM(UNIF, rand_uniform)
BINARY_ELEM(NORM, rand_normal)

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
UNARY_COPY(RESHAPE)

// #undef UNARY_ELEM
// #undef BINARY_ELEM
// #undef UNARY_REDUCE
// #undef UNARY_COPY

template <ade::OPCODE opcode, typename... Args>
void exec (GenericData& out,
	std::vector<GenericData>& data, Args... args)
{
	switch (out.dtype_)
	{
		case DOUBLE:
			Executer<opcode,double,Args...>::exec(out, data, args...);
		break;
		case FLOAT:
			Executer<opcode,float,Args...>::exec(out, data, args...);
		break;
		case INT8:
			Executer<opcode,int8_t,Args...>::exec(out, data, args...);
		break;
		case INT16:
			Executer<opcode,int16_t,Args...>::exec(out, data, args...);
		break;
		case INT32:
			Executer<opcode,int32_t,Args...>::exec(out, data, args...);
		break;
		case INT64:
			Executer<opcode,int64_t,Args...>::exec(out, data, args...);
		break;
		case UINT8:
			Executer<opcode,uint8_t,Args...>::exec(out, data, args...);
		break;
		case UINT16:
			Executer<opcode,uint16_t,Args...>::exec(out, data, args...);
		break;
		case UINT32:
			Executer<opcode,uint32_t,Args...>::exec(out, data, args...);
		break;
		case UINT64:
			Executer<opcode,uint64_t,Args...>::exec(out, data, args...);
		break;
		default:
			util::handle_error("executing bad type");
	}
}

template <typename... Args>
void op_exec (ade::OPCODE opcode, GenericData& out,
	std::vector<GenericData>& data, Args... args)
{
	switch (opcode)
	{
		case ade::ABS:
			exec<ade::ABS, Args...>(out, data, args...);
		break;
		case ade::NEG:
			exec<ade::NEG, Args...>(out, data, args...);
		break;
		case ade::NOT:
			exec<ade::NOT, Args...>(out, data, args...);
		break;
		case ade::SIN:
			exec<ade::SIN, Args...>(out, data, args...);
		break;
		case ade::COS:
			exec<ade::COS, Args...>(out, data, args...);
		break;
		case ade::TAN:
			exec<ade::TAN, Args...>(out, data, args...);
		break;
		case ade::EXP:
			exec<ade::EXP, Args...>(out, data, args...);
		break;
		case ade::LOG:
			exec<ade::LOG, Args...>(out, data, args...);
		break;
		case ade::SQRT:
			exec<ade::SQRT, Args...>(out, data, args...);
		break;
		case ade::ROUND:
			exec<ade::ROUND, Args...>(out, data, args...);
		break;
		case ade::FLIP:
			exec<ade::FLIP, Args...>(out, data, args...);
		break;
		case ade::POW:
			exec<ade::POW, Args...>(out, data, args...);
		break;
		case ade::ADD:
			exec<ade::ADD, Args...>(out, data, args...);
		break;
		case ade::SUB:
			exec<ade::SUB, Args...>(out, data, args...);
		break;
		case ade::MUL:
			exec<ade::MUL, Args...>(out, data, args...);
		break;
		case ade::DIV:
			exec<ade::DIV, Args...>(out, data, args...);
		break;
		case ade::EQ:
			exec<ade::EQ, Args...>(out, data, args...);
		break;
		case ade::NE:
			exec<ade::NE, Args...>(out, data, args...);
		break;
		case ade::LT:
			exec<ade::LT, Args...>(out, data, args...);
		break;
		case ade::GT:
			exec<ade::GT, Args...>(out, data, args...);
		break;
		case ade::BINO:
			exec<ade::BINO, Args...>(out, data, args...);
		break;
		case ade::UNIF:
			exec<ade::UNIF, Args...>(out, data, args...);
		break;
		case ade::NORM:
			exec<ade::NORM, Args...>(out, data, args...);
		break;
		case ade::N_ELEMS:
			exec<ade::N_ELEMS, Args...>(out, data, args...);
		break;
		case ade::N_DIMS:
			exec<ade::N_DIMS, Args...>(out, data, args...);
		break;
		case ade::ARGMAX:
			exec<ade::ARGMAX, Args...>(out, data, args...);
		break;
		case ade::RMAX:
			exec<ade::RMAX, Args...>(out, data, args...);
		break;
		case ade::RSUM:
			exec<ade::RSUM, Args...>(out, data, args...);
		break;
		case ade::MATMUL:
			exec<ade::MATMUL, Args...>(out, data, args...);
		break;
		case ade::PERMUTE:
			exec<ade::PERMUTE, Args...>(out, data, args...);
		break;
		case ade::EXTEND:
			exec<ade::EXTEND, Args...>(out, data, args...);
		break;
		case ade::RESHAPE:
			exec<ade::RESHAPE, Args...>(out, data, args...);
		break;
		default:
			util::handle_error("unknown opcode");
	}
}
