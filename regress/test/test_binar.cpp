#ifndef DISABLE_TF_BINAR_TESTS

#include "regressutil/tf_verify.hpp"

#include "kiln/variable.hpp"
#include "kiln/operators.hpp"
#include "kiln/delta.hpp"


class BINAR_TESTS : public TF_VERIFY
{
public:
	BINAR_TESTS (void)
	{
		TF_VERIFY::to_mem("BINAR");
	}
};


using BINARY_OP = std::function<kiln::Identifier*(kiln::Variable*,kiln::Variable*)>;


void check_binar (OpArgs args, BINARY_OP op)
{
	assert(args.vars_.size() == 5);
	kiln::Variable* input1 = args.vars_[0];
	kiln::Variable* input2 = args.vars_[1];
	kiln::Variable* expectout = args.vars_[2];
	kiln::Variable* expectgrad1 = args.vars_[3];
	kiln::Variable* expectgrad2 = args.vars_[4];

	kiln::Identifier* out = op(input1, input2);
	kiln::Graph::get_global().initialize_all();

	clay::State expectt = expectout->get_state();
	clay::State outt = out->get_state();
	state_check(expectt, outt);

	kiln::Identifier* grad1 = kiln::delta(out, input1);
	clay::State expectgrad1t = expectgrad1->get_state();
	clay::State grad1t = grad1->get_state();
	state_check(expectgrad1t, grad1t);

	kiln::Identifier* grad2 = kiln::delta(out, input2);
	clay::State expectgrad2t = expectgrad2->get_state();
	clay::State grad2t = grad2->get_state();
	state_check(expectgrad2t, grad2t);

	delete input1;
	delete input2;
	delete expectout;
	delete expectgrad1;
	delete expectgrad2;
}


TEST_F(BINAR_TESTS, Add)
{
	check_binar(parse_line("add"),
	[](kiln::Variable* a, kiln::Variable* b) -> kiln::Identifier*
	{
		return kiln::add(a, b);
	});
}


TEST_F(BINAR_TESTS, Sub)
{
	check_binar(parse_line("sub"),
	[](kiln::Variable* a, kiln::Variable* b) -> kiln::Identifier*
	{
		return kiln::sub(a, b);
	});
}


TEST_F(BINAR_TESTS, Mul)
{
	check_binar(parse_line("mul"),
	[](kiln::Variable* a, kiln::Variable* b) -> kiln::Identifier*
	{
		return kiln::mul(a, b);
	});
}


TEST_F(BINAR_TESTS, Div)
{
	check_binar(parse_line("div"),
	[](kiln::Variable* a, kiln::Variable* b) -> kiln::Identifier*
	{
		return kiln::div(a, b);
	});
}


#endif /* DISABLE_TF_BINAR_TESTS */
