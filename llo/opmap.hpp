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

using DataArgsT = std::vector<std::pair<ade::CoordPtrT,GenericData>>;

template <ade::OPCODE OP, typename T>
struct Executer
{
	static void exec (GenericData& out, DataArgsT& data)
	{
		ade::fatalf("cannot %s of type %s", ade::opname(OP).c_str(),
			nametype(get_type<T>()).c_str());
	}
};

template <typename T>
struct Executer<ade::COPY,T>
{
	static void exec (GenericData& out, DataArgsT& data)
	{
		copy((T*) out.data_.get(), out.shape_, VecRef<T>{
			data[0].first,
			(T*) data[0].second.data_.get(),
			data[0].second.shape_,
		});
	}
};

#define UNARY_ELEM(OP, METHOD)template <typename T>\
struct Executer<ade::OP,T> { static void exec\
(GenericData& out, DataArgsT& data)\
{ METHOD((T*) out.data_.get(), VecRef<T>{data[0].first,\
(T*) data[0].second.data_.get(),\
data[0].second.shape_}); } };

UNARY_ELEM(ABS, abs)
UNARY_ELEM(NEG, neg)
UNARY_ELEM(SIN, sin)
UNARY_ELEM(COS, cos)
UNARY_ELEM(TAN, tan)
UNARY_ELEM(EXP, exp)
UNARY_ELEM(LOG, log)
UNARY_ELEM(SQRT, sqrt)
UNARY_ELEM(ROUND, round)

#undef UNARY_ELEM

#define BINARY_ELEM(OP, METHOD)template <typename T>\
struct Executer<ade::OP,T> { static void exec\
(GenericData& out, DataArgsT& data)\
{ METHOD((T*) out.data_.get(), VecRef<T>{data[0].first,\
(T*) data[0].second.data_.get(), data[0].second.shape_},\
VecRef<T>{data[1].first, (T*) data[1].second.data_.get(),\
data[1].second.shape_}); } };

BINARY_ELEM(POW, pow)
BINARY_ELEM(SUB, sub)
BINARY_ELEM(DIV, div)
BINARY_ELEM(EQ, eq)
BINARY_ELEM(NE, neq)
BINARY_ELEM(LT, lt)
BINARY_ELEM(GT, gt)

template <typename T>
struct Executer<ade::RAND_BINO,T>
{
	static void exec (GenericData& out, DataArgsT& data)
	{
		rand_binom((T*) out.data_.get(),
			VecRef<T>{data[0].first,
				(T*) data[0].second.data_.get(),
				data[0].second.shape_},
			VecRef<double>{data[1].first,
				(double*) data[1].second.data_.get(),
				data[1].second.shape_});
	}
};

BINARY_ELEM(RAND_UNIF, rand_uniform)
BINARY_ELEM(RAND_NORM, rand_normal)

#undef BINARY_ELEM

#define NARY_ELEM(OP, METHOD)template <typename T>\
struct Executer<ade::OP,T> { static void exec\
(GenericData& out, DataArgsT& data)\
{ std::vector<VecRef<T>> args(data.size()); \
std::transform(data.begin(), data.end(), args.begin(), \
[](std::pair<ade::CoordPtrT,GenericData>& gd) {\
return VecRef<T>{gd.first, (T*) gd.second.data_.get(),\
gd.second.shape_}; }); METHOD((T*) out.data_.get(), out.shape_, args); } };

NARY_ELEM(ADD, add)
NARY_ELEM(MUL, mul)
NARY_ELEM(MIN, min)
NARY_ELEM(MAX, max)

#undef NARY_ELEM

template <ade::OPCODE OP>
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
			ade::fatal("executing bad type");
	}
}

void op_exec (ade::OPCODE opcode, GenericData& out, DataArgsT& data);

}

#endif // LLO_OPMAP_HPP
