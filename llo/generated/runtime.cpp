#include "llo/generated/api.hpp"

#ifdef _GENERATED_RT_HPP

namespace age
{

static std::unordered_map<std::string,_GENERATED_OPCODES> code2name =
{
	{"RAND_NORM",RAND_NORM},
	{"MAX",MAX},
	{"LT",LT},
	{"POW",POW},
	{"SUB",SUB},
	{"MIN",MIN},
	{"ABS",ABS},
	{"EXP",EXP},
	{"DIV",DIV},
	{"GT",GT},
	{"COS",COS},
	{"LOG",LOG},
	{"RAND_BINO",RAND_BINO},
	{"TAN",TAN},
	{"NEG",NEG},
	{"NEQ",NEQ},
	{"EQ",EQ},
	{"PROD",PROD},
	{"RAND_UNIF",RAND_UNIF},
	{"ROUND",ROUND},
	{"SIN",SIN},
	{"SQRT",SQRT},
	{"SUM",SUM},
};

ade::Opcode sum_opcode (void)
{
	return ade::Opcode{"SUM",SUM};
}

ade::Opcode prod_opcode (void)
{
	return ade::Opcode{"PROD",PROD};
}

_GENERATED_OPCODES nameop (std::string name)
{
	return code2name.find(name)->second;
}

std::string opname (_GENERATED_OPCODES code)
{
	switch (code)
	{
		case EXP:
		{
			return "EXP";
		}
		case SUB:
		{
			return "SUB";
		}
		case NEQ:
		{
			return "NEQ";
		}
		case SQRT:
		{
			return "SQRT";
		}
		case LT:
		{
			return "LT";
		}
		case GT:
		{
			return "GT";
		}
		case ROUND:
		{
			return "ROUND";
		}
		case EQ:
		{
			return "EQ";
		}
		case TAN:
		{
			return "TAN";
		}
		case NEG:
		{
			return "NEG";
		}
		case COS:
		{
			return "COS";
		}
		case SIN:
		{
			return "SIN";
		}
		case PROD:
		{
			return "PROD";
		}
		case LOG:
		{
			return "LOG";
		}
		case DIV:
		{
			return "DIV";
		}
		case MIN:
		{
			return "MIN";
		}
		case MAX:
		{
			return "MAX";
		}
		case SUM:
		{
			return "SUM";
		}
		case POW:
		{
			return "POW";
		}
		case ABS:
		{
			return "ABS";
		}
		case RAND_BINO:
		{
			return "RAND_BINO";
		}
		case RAND_NORM:
		{
			return "RAND_NORM";
		}
		case RAND_UNIF:
		{
			return "RAND_UNIF";
		}
	}
}

ade::Tensorptr grad_rule (size_t code,TensT args,size_t idx)
{
	switch (code)
	{
		case EXP:
		{
			return exp(args[0]);
		}
		case SUB:
		{
			return idx == 0 ? data(1,args[0]->shape()) : neg(data(1,args[0]->shape()));
		}
		case NEQ:
		{
			return data(0,args[0]->shape());
		}
		case SQRT:
		{
			return div(data(1,args[0]->shape()),mul(data(2,args[0]->shape()),sqrt(args[0])));
		}
		case LT:
		{
			return data(0,args[0]->shape());
		}
		case GT:
		{
			return data(0,args[0]->shape());
		}
		case ROUND:
		{
			return data(1,args[0]->shape());
		}
		case EQ:
		{
			return data(0,args[0]->shape());
		}
		case TAN:
		{
			return div(data(1,args[0]->shape()),pow(cos(args[0]),data(2,args[0]->shape())));
		}
		case NEG:
		{
			return neg(data(1,args[0]->shape()));
		}
		case COS:
		{
			return neg(sin(args[0]));
		}
		case SIN:
		{
			return cos(args[0]);
		}
		case PROD:
		{
			return llo::grad_prod(idx,args);
		}
		case LOG:
		{
			return div(data(1,args[0]->shape()),args[0]);
		}
		case DIV:
		{
			return idx == 0 ? div(data(1,args[0]->shape()),args[1]) : div(neg(args[0]),pow(args[1],data(2,args[0]->shape())));
		}
		case MIN:
		{
			return llo::grad_min(idx,args);
		}
		case MAX:
		{
			return llo::grad_max(idx,args);
		}
		case SUM:
		{
			return data(1,args[0]->shape());
		}
		case POW:
		{
			return idx == 0 ? mul(args[1],pow(args[0],sub(args[1],data(1,args[0]->shape())))) : mul(log(args[0]),pow(args[0],args[1]));
		}
		case ABS:
		{
			return div(args[0],abs(args[0]));
		}
		case RAND_BINO:
		{
			return data(0,args[0]->shape());
		}
		case RAND_NORM:
		{
			return data(0,args[0]->shape());
		}
		case RAND_UNIF:
		{
			return data(0,args[0]->shape());
		}
	}
}

}

#endif
