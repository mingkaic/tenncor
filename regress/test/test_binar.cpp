#ifndef DISABLE_TF_BINAR_TESTS

#include "regress_util/tf_verify.hpp"

#include "wire/variable.hpp"
#include "wire/operators.hpp"


class BINAR_TESTS : public TF_VERIFY
{
public:
	BINAR_TESTS (void)
	{
		TF_VERIFY::to_mem("BINAR");
	}
};


using BINARY_OP = std::function<wire::Identifier*(wire::Variable*,wire::Variable*)>;


void check_binar (OpArgs args, BINARY_OP op)
{
	assert(args.vars_.size() == 5);
	wire::Variable* input1 = args.vars_[0];
	wire::Variable* input2 = args.vars_[1];
	wire::Variable* expectout = args.vars_[2];
	wire::Variable* expectgrad1 = args.vars_[3];
	wire::Variable* expectgrad2 = args.vars_[4];

	wire::Identifier* out = op(input1, input2);
	wire::Graph::get_global().initialize_all();

	clay::State expectt = expectout->get_state();
	clay::State outt = out->get_state();
	state_check(expectt, outt);

	wire::Identifier* grad1 = out->derive(input1);
	clay::State expectgrad1t = expectgrad1->get_state();
	clay::State grad1t = grad1->get_state();
	state_check(expectgrad1t, grad1t);

	wire::Identifier* grad2 = out->derive(input2);
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
	[](wire::Variable* a, wire::Variable* b) -> wire::Identifier*
	{
		return wire::add(a, b);
	});
}


TEST_F(BINAR_TESTS, Sub)
{
	check_binar(parse_line("sub"),
	[](wire::Variable* a, wire::Variable* b) -> wire::Identifier*
	{
		return wire::sub(a, b);
	});
}


TEST_F(BINAR_TESTS, Mul)
{
	check_binar(parse_line("mul"),
	[](wire::Variable* a, wire::Variable* b) -> wire::Identifier*
	{
		return wire::mul(a, b);
	});
}


TEST_F(BINAR_TESTS, Div)
{
	check_binar(parse_line("div"),
	[](wire::Variable* a, wire::Variable* b) -> wire::Identifier*
	{
		return wire::div(a, b);
	});
}


#endif /* DISABLE_TF_BINAR_TESTS */
