#ifndef DISABLE_TF_UNAR_TESTS

#include "tests/functional/include/tf_verify.hpp"

#include "graph/leaf/variable.hpp"
#include "operate/operations.hpp"


class UNAR : public TF_VERIFY
{
public:
	UNAR (void)
	{
		TF_VERIFY::to_mem("UNAR");
	}
};


void check_unar (op_args args, std::function<nnet::varptr(nnet::varptr)> op)
{
	nnet::variable* input = args.vars_[0];
	nnet::variable* expectout = args.vars_[1];
	nnet::variable* expectgrad = args.vars_[2];

	nnet::varptr out = op(nnet::varptr(input));
	input->initialize();
	expectout->initialize();
	expectgrad->initialize();

	nnet::tensor* expectt = expectout->get_tensor();
	nnet::tensor* outt = out->get_tensor();
	tensor_check(expectt, outt);

	nnet::varptr grad = out->derive(input);
	nnet::tensor* expectgradt = expectgrad->get_tensor();
	nnet::tensor* gradt = grad->get_tensor();
	tensor_check(expectgradt, gradt);

	delete input;
	delete expectout;
	delete expectgrad;
}


TEST_F(UNAR, Abs)
{
	check_unar(parse_line("abs"), [](nnet::varptr in) {
		return nnet::abs(in);
	});
}


TEST_F(UNAR, Neg)
{
	check_unar(parse_line("neg"), [](nnet::varptr in) {
		return -in;
	});
}


TEST_F(UNAR, Sin)
{
	check_unar(parse_line("sin"), [](nnet::varptr in) {
		return nnet::sin(in);
	});
}


TEST_F(UNAR, Cos)
{
	check_unar(parse_line("cos"), [](nnet::varptr in) {
		return nnet::cos(in);
	});
}


TEST_F(UNAR, Tan)
{
	check_unar(parse_line("tan"), [](nnet::varptr in) {
		return nnet::tan(in);
	});
}


TEST_F(UNAR, Exp)
{
	check_unar(parse_line("exp"), [](nnet::varptr in) {
		return nnet::exp(in);
	});
}


TEST_F(UNAR, Log) // Precision problem
{
	check_unar(parse_line("log"), [](nnet::varptr in) {
		return nnet::log(in);
	});
}


TEST_F(UNAR, Sqrt)
{
	check_unar(parse_line("sqrt"), [](nnet::varptr in) {
		return nnet::sqrt(in);
	});
}


#endif /* DISABLE_TF_UNAR_TESTS */
