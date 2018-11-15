#include "ade/functor.hpp"
#include "llo/generated/code.hpp"
#include "llo/data.hpp"

#ifndef _GENERATED_OPERA_HPP
#define _GENERATED_OPERA_HPP

namespace age
{

template <typename T>
void typed_exec (_GENERATED_OPCODE opcode,
	char* out, ade::Shape shape, llo::DataArgsT& in)
{
	switch (opcode)
	{
		case COS:
			llo::cos((T*)out,llo::to_ref<T>(in[0])); break;
		case LOG:
			llo::log((T*)out,llo::to_ref<T>(in[0])); break;
		case MIN:
			llo::min((T*)out,shape,llo::to_refs<T>(in)); break;
		case NEG:
			llo::neg((T*)out,llo::to_ref<T>(in[0])); break;
		case SUM:
			llo::add((T*)out,shape,llo::to_refs<T>(in)); break;
		case EXP:
			llo::exp((T*)out,llo::to_ref<T>(in[0])); break;
		case DIV:
			llo::div((T*)out,shape,llo::to_ref<T>(in[0]),llo::to_ref<T>(in[1])); break;
		case SIN:
			llo::sin((T*)out,llo::to_ref<T>(in[0])); break;
		case SUB:
			llo::sub((T*)out,shape,llo::to_ref<T>(in[0]),llo::to_ref<T>(in[1])); break;
		case RAND_NORM:
			llo::rand_normal((T*)out,shape,llo::to_ref<T>(in[0]),llo::to_ref<T>(in[1])); break;
		case POW:
			llo::pow((T*)out,shape,llo::to_ref<T>(in[0]),llo::to_ref<T>(in[1])); break;
		case ROUND:
			llo::round((T*)out,llo::to_ref<T>(in[0])); break;
		case LT:
			llo::lt((T*)out,shape,llo::to_ref<T>(in[0]),llo::to_ref<T>(in[1])); break;
		case TAN:
			llo::tan((T*)out,llo::to_ref<T>(in[0])); break;
		case NEQ:
			llo::neq((T*)out,shape,llo::to_ref<T>(in[0]),llo::to_ref<T>(in[1])); break;
		case GT:
			llo::gt((T*)out,shape,llo::to_ref<T>(in[0]),llo::to_ref<T>(in[1])); break;
		case ABS:
			llo::abs((T*)out,llo::to_ref<T>(in[0])); break;
		case EQ:
			llo::eq((T*)out,shape,llo::to_ref<T>(in[0]),llo::to_ref<T>(in[1])); break;
		case MAX:
			llo::max((T*)out,shape,llo::to_refs<T>(in)); break;
		case SQRT:
			llo::sqrt((T*)out,llo::to_ref<T>(in[0])); break;
		case RAND_UNIF:
			llo::rand_uniform((T*)out,shape,llo::to_ref<T>(in[0]),llo::to_ref<T>(in[1])); break;
		case PROD:
			llo::mul((T*)out,shape,llo::to_refs<T>(in)); break;
		case RAND_BINO:
			llo::rand_binom((T*)out,shape,llo::to_ref<T>(in[0]),llo::to_ref<double>(in[1])); break;
		default: err::fatal("unknown opcode");
	}
}

void op_exec (_GENERATED_OPCODE opcode, _GENERATED_DTYPE dtype,
	char* out, ade::Shape shape, llo::DataArgsT& in);

}

#endif // _GENERATED_OPERA_HPP
