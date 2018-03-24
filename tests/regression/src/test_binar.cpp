#ifndef DISABLE_TF_BINAR_TESTS

#include "tests/regression/include/tf_verify.hpp"

#include "graph/leaf/variable.hpp"
#include "operate/operations.hpp"


class BINAR_TESTS : public TF_VERIFY
{
public:
	BINAR_TESTS (void)
	{
		TF_VERIFY::to_mem("BINAR");
	}
};


void check_binar (op_args args, std::function<nnet::varptr(nnet::varptr,nnet::varptr)> op)
{
	assert(args.vars_.size() == 5);
	nnet::variable* input1 = args.vars_[0];
	nnet::variable* input2 = args.vars_[1];
	nnet::variable* expectout = args.vars_[2];
	nnet::variable* expectgrad1 = args.vars_[3];
	nnet::variable* expectgrad2 = args.vars_[4];

	nnet::varptr out = op(nnet::varptr(input1), nnet::varptr(input2));
	input1->initialize();
	input2->initialize();
	expectout->initialize();
	expectgrad1->initialize();
	expectgrad2->initialize();

	nnet::tensor* expectt = expectout->get_tensor();
	nnet::tensor* outt = out->get_tensor();
	tensor_check(expectt, outt);

	nnet::varptr grad1 = out->derive(input1);
	nnet::tensor* expectgrad1t = expectgrad1->get_tensor();
	nnet::tensor* grad1t = grad1->get_tensor();
	tensor_check(expectgrad1t, grad1t);

	nnet::varptr grad2 = out->derive(input2);
	nnet::tensor* expectgrad2t = expectgrad2->get_tensor();
	nnet::tensor* grad2t = grad2->get_tensor();
	tensor_check(expectgrad2t, grad2t);

	delete input1;
	delete input2;
	delete expectout;
	delete expectgrad1;
	delete expectgrad2;
}


TEST_F(BINAR_TESTS, Add)
{
	check_binar(parse_line("add"), [](nnet::varptr a, nnet::varptr b)
	{
		return a + b;
	});
}


TEST_F(BINAR_TESTS, Sub)
{
	check_binar(parse_line("sub"), [](nnet::varptr a, nnet::varptr b)
	{
		return a - b;
	});
}


TEST_F(BINAR_TESTS, Mul)
{
	check_binar(parse_line("mul"), [](nnet::varptr a, nnet::varptr b)
	{
		return a * b;
	});
}


TEST_F(BINAR_TESTS, Div)
{
	check_binar(parse_line("div"), [](nnet::varptr a, nnet::varptr b)
	{
		return a / b;
	});
}


#endif /* DISABLE_TF_BINAR_TESTS */
