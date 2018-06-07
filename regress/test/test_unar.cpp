#ifndef DISABLE_TF_UNAR_TESTS

#include "regressutil/tf_verify.hpp"

#include "wire/variable.hpp"
#include "wire/operators.hpp"
#include "wire/delta.hpp"


class UNAR_TESTS : public TF_VERIFY
{
public:
	UNAR_TESTS (void)
	{
		TF_VERIFY::to_mem("UNAR");
	}
};


using UNARY_OP = std::function<wire::Identifier*(wire::Variable*)>;


void check_unar (OpArgs args, UNARY_OP op)
{
	assert(args.vars_.size() == 3);
	wire::Variable* input = args.vars_[0];
	wire::Variable* expectout = args.vars_[1];
	wire::Variable* expectgrad = args.vars_[2];

	wire::Identifier* out = op(input);
	wire::Graph::get_global().initialize_all();

	clay::State expectt = expectout->get_state();
	clay::State outt = out->get_state();
	state_check(expectt, outt);

	wire::Identifier* grad = wire::delta(out, input);
	clay::State expectgradt = expectgrad->get_state();
	clay::State gradt = grad->get_state();
	state_check(expectgradt, gradt);

	delete input;
	delete expectout;
	delete expectgrad;
}


TEST_F(UNAR_TESTS, Abs)
{
	check_unar(parse_line("abs"),
	[](wire::Variable* in) -> wire::Identifier*
	{
		return wire::abs(in);
	});
}


TEST_F(UNAR_TESTS, Neg)
{
	check_unar(parse_line("neg"),
	[](wire::Variable* in) -> wire::Identifier*
	{
		return wire::neg(in);
	});
}


TEST_F(UNAR_TESTS, Sin)
{
	check_unar(parse_line("sin"),
	[](wire::Variable* in) -> wire::Identifier*
	{
		return wire::sin(in);
	});
}


TEST_F(UNAR_TESTS, Cos)
{
	check_unar(parse_line("cos"),
	[](wire::Variable* in) -> wire::Identifier*
	{
		return wire::cos(in);
	});
}


TEST_F(UNAR_TESTS, Tan)
{
	check_unar(parse_line("tan"),
	[](wire::Variable* in) -> wire::Identifier*
	{
		return wire::tan(in);
	});
}


TEST_F(UNAR_TESTS, Exp)
{
	check_unar(parse_line("exp"),
	[](wire::Variable* in) -> wire::Identifier*
	{
		return wire::exp(in);
	});
}


TEST_F(UNAR_TESTS, Log) // Precision problem
{
	check_unar(parse_line("log"),
	[](wire::Variable* in) -> wire::Identifier*
	{
		return wire::log(in);
	});
}


TEST_F(UNAR_TESTS, Sqrt)
{
	check_unar(parse_line("sqrt"),
	[](wire::Variable* in) -> wire::Identifier*
	{
		return wire::sqrt(in);
	});
}


#endif /* DISABLE_TF_UNAR_TESTS */
