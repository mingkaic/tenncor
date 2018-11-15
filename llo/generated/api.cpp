#include "llo/generated/code.hpp"
#include "llo/generated/api.hpp"
#include "llo/helper.hpp"

#ifdef _GENERATED_API_HPP

namespace age
{

ade::Tensorptr abs (ade::Tensorptr arg1)
{
	return ade::Functor::get(ade::Opcode{"ABS",ABS},{{ade::identity,arg1}});
}

ade::Tensorptr neg (ade::Tensorptr arg1)
{
	return ade::Functor::get(ade::Opcode{"NEG",NEG},{{ade::identity,arg1}});
}

ade::Tensorptr sin (ade::Tensorptr arg1)
{
	return ade::Functor::get(ade::Opcode{"SIN",SIN},{{ade::identity,arg1}});
}

ade::Tensorptr cos (ade::Tensorptr arg1)
{
	return ade::Functor::get(ade::Opcode{"COS",COS},{{ade::identity,arg1}});
}

ade::Tensorptr tan (ade::Tensorptr arg1)
{
	return ade::Functor::get(ade::Opcode{"TAN",TAN},{{ade::identity,arg1}});
}

ade::Tensorptr exp (ade::Tensorptr arg1)
{
	return ade::Functor::get(ade::Opcode{"EXP",EXP},{{ade::identity,arg1}});
}

ade::Tensorptr log (ade::Tensorptr arg1)
{
	return ade::Functor::get(ade::Opcode{"LOG",LOG},{{ade::identity,arg1}});
}

ade::Tensorptr sqrt (ade::Tensorptr arg1)
{
	return ade::Functor::get(ade::Opcode{"SQRT",SQRT},{{ade::identity,arg1}});
}

ade::Tensorptr round (ade::Tensorptr arg1)
{
	return ade::Functor::get(ade::Opcode{"ROUND",ROUND},{{ade::identity,arg1}});
}

ade::Tensorptr flip (ade::Tensorptr arg1, uint8_t arg2)
{
	return ade::Functor::get(ade::Opcode{"SUM",SUM},{{ade::flip(arg2),arg1}});
}

ade::Tensorptr pow (ade::Tensorptr arg1, ade::Tensorptr arg2)
{
	return ade::Functor::get(ade::Opcode{"POW",POW},{{ade::identity,arg1},{ade::identity,arg2}});
}

ade::Tensorptr add (ade::Tensorptr arg1, ade::Tensorptr arg2)
{
	return ade::Functor::get(ade::Opcode{"SUM",SUM},{{ade::identity,arg1},{ade::identity,arg2}});
}

ade::Tensorptr sub (ade::Tensorptr arg1, ade::Tensorptr arg2)
{
	return ade::Functor::get(ade::Opcode{"SUB",SUB},{{ade::identity,arg1},{ade::identity,arg2}});
}

ade::Tensorptr mul (ade::Tensorptr arg1, ade::Tensorptr arg2)
{
	return ade::Functor::get(ade::Opcode{"PROD",PROD},{{ade::identity,arg1},{ade::identity,arg2}});
}

ade::Tensorptr div (ade::Tensorptr arg1, ade::Tensorptr arg2)
{
	return ade::Functor::get(ade::Opcode{"DIV",DIV},{{ade::identity,arg1},{ade::identity,arg2}});
}

ade::Tensorptr eq (ade::Tensorptr arg1, ade::Tensorptr arg2)
{
	return ade::Functor::get(ade::Opcode{"EQ",EQ},{{ade::identity,arg1},{ade::identity,arg2}});
}

ade::Tensorptr neq (ade::Tensorptr arg1, ade::Tensorptr arg2)
{
	return ade::Functor::get(ade::Opcode{"NEQ",NEQ},{{ade::identity,arg1},{ade::identity,arg2}});
}

ade::Tensorptr lt (ade::Tensorptr arg1, ade::Tensorptr arg2)
{
	return ade::Functor::get(ade::Opcode{"LT",LT},{{ade::identity,arg1},{ade::identity,arg2}});
}

ade::Tensorptr gt (ade::Tensorptr arg1, ade::Tensorptr arg2)
{
	return ade::Functor::get(ade::Opcode{"GT",GT},{{ade::identity,arg1},{ade::identity,arg2}});
}

ade::Tensorptr rand_bino (ade::Tensorptr arg1, ade::Tensorptr arg2)
{
	return ade::Functor::get(ade::Opcode{"RAND_BINO",RAND_BINO},{{ade::identity,arg1},{ade::identity,arg2}});
}

ade::Tensorptr rand_unif (ade::Tensorptr arg1, ade::Tensorptr arg2)
{
	return ade::Functor::get(ade::Opcode{"RAND_UNIF",RAND_UNIF},{{ade::identity,arg1},{ade::identity,arg2}});
}

ade::Tensorptr rand_norm (ade::Tensorptr arg1, ade::Tensorptr arg2)
{
	return ade::Functor::get(ade::Opcode{"RAND_NORM",RAND_NORM},{{ade::identity,arg1},{ade::identity,arg2}});
}

ade::Tensorptr sum (age::TensT arg1)
{
	return ade::Functor::get(ade::Opcode{"SUM",SUM},age::to_args(arg1));
}

ade::Tensorptr prod (age::TensT arg1)
{
	return ade::Functor::get(ade::Opcode{"PROD",PROD},age::to_args(arg1));
}

ade::Tensorptr min (age::TensT arg1)
{
	return ade::Functor::get(ade::Opcode{"MIN",MIN},age::to_args(arg1));
}

ade::Tensorptr max (age::TensT arg1)
{
	return ade::Functor::get(ade::Opcode{"MAX",MAX},age::to_args(arg1));
}

ade::Tensorptr reduce_sum (ade::Tensorptr arg1, uint8_t arg2)
{
	return ade::Functor::get(ade::Opcode{"SUM",SUM},{{llo::reduce(arg2,arg1->shape()),arg1}});
}

ade::Tensorptr reduce_min (ade::Tensorptr arg1, uint8_t arg2)
{
	return ade::Functor::get(ade::Opcode{"MIN",MIN},{{llo::reduce(arg2,arg1->shape()),arg1}});
}

ade::Tensorptr reduce_max (ade::Tensorptr arg1, uint8_t arg2)
{
	return ade::Functor::get(ade::Opcode{"MAX",MAX},{{llo::reduce(arg2,arg1->shape()),arg1}});
}

ade::Tensorptr permute (ade::Tensorptr arg1, std::vector<uint8_t> arg2)
{
	return ade::Functor::get(ade::Opcode{"SUM",SUM},{{ade::permute(arg2),arg1}});
}

ade::Tensorptr extend (ade::Tensorptr arg1, uint8_t arg2, std::vector<uint8_t> arg3)
{
	return ade::Functor::get(ade::Opcode{"SUM",SUM},{{ade::extend(arg2,arg3),arg1}});
}

ade::Tensorptr reduce_sum (ade::Tensorptr arg1)
{
	return reduce_sum(arg1,0);
}

ade::Tensorptr reduce_min (ade::Tensorptr arg1)
{
	return reduce_min(arg1,0);
}

ade::Tensorptr reduce_max (ade::Tensorptr arg1)
{
	return reduce_max(arg1,0);
}

ade::Tensorptr matmul (ade::Tensorptr arg1, ade::Tensorptr arg2)
{
	return reduce_sum(mul(permute(extend(arg1,2,{arg2->shape().at(0)}),{2,1,0}),permute(extend(arg2,2,{arg1->shape().at(1)}),{0,2,1})),2);
}

ade::Tensorptr convolute (ade::Tensorptr arg1, ade::Tensorptr arg2)
{
	return ade::Tensorptr(nullptr);
}


}

#endif
