#ifndef DISABLE_TF_UNAR_TESTS

#include "regressutil/tf_verify.hpp"

#include "kiln/variable.hpp"
#include "kiln/operators.hpp"
#include "kiln/delta.hpp"


class UNAR_TESTS : public TF_VERIFY
{
public:
	UNAR_TESTS (void)
	{
		TF_VERIFY::to_mem("UNAR");
	}
};


using UNARY_OP = std::function<kiln::Identifier*(kiln::Variable*)>;


void check_unar (OpArgs args, UNARY_OP op)
{
	assert(args.vars_.size() == 3);
	kiln::Variable* input = args.vars_[0];
	kiln::Variable* expectout = args.vars_[1];
	kiln::Variable* expectgrad = args.vars_[2];

	kiln::Identifier* out = op(input);
	kiln::Graph::get_global().initialize_all();

	clay::State expectt = expectout->get_state();
	clay::State outt = out->get_state();
	state_check(expectt, outt);

	kiln::Identifier* grad = kiln::delta(out, input);
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
	[](kiln::Variable* in) -> kiln::Identifier*
	{
		return kiln::abs(in);
	});
}


TEST_F(UNAR_TESTS, Neg)
{
	check_unar(parse_line("neg"),
	[](kiln::Variable* in) -> kiln::Identifier*
	{
		return kiln::neg(in);
	});
}


TEST_F(UNAR_TESTS, Sin)
{
	check_unar(parse_line("sin"),
	[](kiln::Variable* in) -> kiln::Identifier*
	{
		return kiln::sin(in);
	});
}


TEST_F(UNAR_TESTS, Cos)
{
	check_unar(parse_line("cos"),
	[](kiln::Variable* in) -> kiln::Identifier*
	{
		return kiln::cos(in);
	});
}


TEST_F(UNAR_TESTS, Tan)
{
	check_unar(parse_line("tan"),
	[](kiln::Variable* in) -> kiln::Identifier*
	{
		return kiln::tan(in);
	});
}


TEST_F(UNAR_TESTS, Exp)
{
	check_unar(parse_line("exp"),
	[](kiln::Variable* in) -> kiln::Identifier*
	{
		return kiln::exp(in);
	});
}


TEST_F(UNAR_TESTS, Log) // Precision problem
{
	check_unar(parse_line("log"),
	[](kiln::Variable* in) -> kiln::Identifier*
	{
		return kiln::log(in);
	});
}


TEST_F(UNAR_TESTS, Sqrt)
{
	check_unar(parse_line("sqrt"),
	[](kiln::Variable* in) -> kiln::Identifier*
	{
		return kiln::sqrt(in);
	});
}


#endif /* DISABLE_TF_UNAR_TESTS */
