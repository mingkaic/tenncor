#include "llo/generated/api.hpp"

#ifdef _GENERATED_RT_HPP

namespace age
{

static std::unordered_map<std::string,_GENERATED_OPCODES> code2name =
{
	{"EXP",EXP},
	{"LOG",LOG},
	{"NEG",NEG},
	{"RAND_UNIF",RAND_UNIF},
	{"MIN",MIN},
	{"MAX",MAX},
	{"COS",COS},
	{"DIV",DIV},
	{"ABS",ABS},
	{"ROUND",ROUND},
	{"GT",GT},
	{"NEQ",NEQ},
	{"EQ",EQ},
	{"POW",POW},
	{"SUM",SUM},
	{"LT",LT},
	{"PROD",PROD},
	{"RAND_BINO",RAND_BINO},
	{"RAND_NORM",RAND_NORM},
	{"SIN",SIN},
	{"SQRT",SQRT},
	{"SUB",SUB},
	{"TAN",TAN},
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
		case TAN:
		{
			return "TAN";
		}
		case SUM:
		{
			return "SUM";
		}
		case RAND_UNIF:
		{
			return "RAND_UNIF";
		}
		case RAND_NORM:
		{
			return "RAND_NORM";
		}
		case RAND_BINO:
		{
			return "RAND_BINO";
		}
		case PROD:
		{
			return "PROD";
		}
		case SQRT:
		{
			return "SQRT";
		}
		case MIN:
		{
			return "MIN";
		}
		case POW:
		{
			return "POW";
		}
		case MAX:
		{
			return "MAX";
		}
		case LOG:
		{
			return "LOG";
		}
		case SIN:
		{
			return "SIN";
		}
		case NEG:
		{
			return "NEG";
		}
		case ROUND:
		{
			return "ROUND";
		}
		case EXP:
		{
			return "EXP";
		}
		case LT:
		{
			return "LT";
		}
		case EQ:
		{
			return "EQ";
		}
		case NEQ:
		{
			return "NEQ";
		}
		case GT:
		{
			return "GT";
		}
		case DIV:
		{
			return "DIV";
		}
		case SUB:
		{
			return "SUB";
		}
		case COS:
		{
			return "COS";
		}
		case ABS:
		{
			return "ABS";
		}
	}
}

ade::Tensorptr grad_rule (size_t code, TensT args, size_t idx)
{
	switch (code)
	{
		case TAN:
		{
			return div(data(1,args[0]->shape()),pow(cos(args[0]),data(2,args[0]->shape())));
		}
		case SUM:
		{
			return data(1,args[0]->shape());
		}
		case RAND_UNIF:
		{
			return data(0,args[0]->shape());
		}
		case RAND_NORM:
		{
			return data(0,args[0]->shape());
		}
		case RAND_BINO:
		{
			return data(0,args[0]->shape());
		}
		case PROD:
		{
			return llo::grad_prod(idx,args);
		}
		case SQRT:
		{
			return div(data(1,args[0]->shape()),mul(data(2,args[0]->shape()),sqrt(args[0])));
		}
		case MIN:
		{
			return llo::grad_min(idx,args);
		}
		case POW:
		{
			return idx == 0 ? mul(args[1],pow(args[0],sub(args[1],data(1,args[0]->shape())))) : mul(log(args[0]),pow(args[0],args[1]));
		}
		case MAX:
		{
			return llo::grad_max(idx,args);
		}
		case LOG:
		{
			return div(data(1,args[0]->shape()),args[0]);
		}
		case SIN:
		{
			return cos(args[0]);
		}
		case NEG:
		{
			return neg(data(1,args[0]->shape()));
		}
		case ROUND:
		{
			return data(1,args[0]->shape());
		}
		case EXP:
		{
			return exp(args[0]);
		}
		case LT:
		{
			return data(0,args[0]->shape());
		}
		case EQ:
		{
			return data(0,args[0]->shape());
		}
		case NEQ:
		{
			return data(0,args[0]->shape());
		}
		case GT:
		{
			return data(0,args[0]->shape());
		}
		case DIV:
		{
			return idx == 0 ? div(data(1,args[0]->shape()),args[1]) : div(neg(args[0]),pow(args[1],data(2,args[0]->shape())));
		}
		case SUB:
		{
			return idx == 0 ? data(1,args[0]->shape()) : neg(data(1,args[0]->shape()));
		}
		case COS:
		{
			return neg(sin(args[0]));
		}
		case ABS:
		{
			return div(args[0],abs(args[0]));
		}
	}
}

}

#endif
