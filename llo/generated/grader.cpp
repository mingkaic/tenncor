#include "llo/generated/code.hpp"
#include "llo/generated/api.hpp"
#include "llo/generated/grader.hpp"

#ifdef _GENERATED_GRADER_HPP

namespace age
{

ade::Tensorptr grad_rule (size_t code,TensT args,size_t idx)
{
	switch (code)
	{
		case COS: return neg(sin(args[0]));
		case LOG: return div(data(1,args[0]->shape()),args[0]);
		case MIN: return llo::grad_min(idx,args);
		case NEG: return neg(data(1,args[0]->shape()));
		case SUM: return data(1,args[0]->shape());
		case EXP: return exp(args[0]);
		case DIV: return idx == 0 ? div(data(1,args[0]->shape()),args[1]) : div(neg(args[0]),pow(args[1],data(2,args[0]->shape())));
		case SIN: return cos(args[0]);
		case SUB: return idx == 0 ? data(1,args[0]->shape()) : neg(data(1,args[0]->shape()));
		case RAND_NORM: return data(0,args[0]->shape());
		case POW: return idx == 0 ? mul(args[1],pow(args[0],sub(args[1],data(1,args[0]->shape())))) : mul(log(args[0]),pow(args[0],args[1]));
		case ROUND: return data(1,args[0]->shape());
		case LT: return data(0,args[0]->shape());
		case TAN: return div(data(1,args[0]->shape()),pow(cos(args[0]),data(2,args[0]->shape())));
		case NEQ: return data(0,args[0]->shape());
		case GT: return data(0,args[0]->shape());
		case ABS: return div(args[0],abs(args[0]));
		case EQ: return data(0,args[0]->shape());
		case MAX: return llo::grad_max(idx,args);
		case SQRT: return div(data(1,args[0]->shape()),mul(data(2,args[0]->shape()),sqrt(args[0])));
		case RAND_UNIF: return data(0,args[0]->shape());
		case PROD: return llo::grad_prod(idx,args);
		case RAND_BINO: return data(0,args[0]->shape());
        default: err::fatal("no gradient rule for unknown opcode");
    }
}

}

#endif
