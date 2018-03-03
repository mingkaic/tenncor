#ifndef DISABLE_TF_UNAR_SCALR_TESTS

#include "tests/functional/include/tf_verify.hpp"

#include "graph/leaf/variable.hpp"
#include "operations/operations.hpp"

class UNAR_SCALR : public TF_VERIFY
{
public:
	UNAR_SCALR (void)
	{
		TF_VERIFY::to_mem("UNAR_SCALR");
	}
};


void check_unar_scalr (op_args args, std::function<nnet::varptr(nnet::varptr,std::vector<double>)> op)
{
	nnet::variable* input = args.vars_[0];
	nnet::variable* expectout = args.vars_[1];
	nnet::variable* expectgrad = args.vars_[2];

	nnet::varptr out = op(nnet::varptr(input), args.params_);
	input->initialize();
	expectout->initialize();
	expectgrad->initialize();

	nnet::tensor* expectt = expectout->get_tensor();
	nnet::tensor* outt = out->get_tensor();
	tensor_check(expectt, outt);

	// nnet::varptr grad = out->derive(input);
	// nnet::tensor* expectgradt = expectgrad->get_tensor();
	// nnet::tensor* gradt = grad->get_tensor();
	// tensor_check(expectgradt, gradt);

	delete input;
	delete expectout;
	delete expectgrad;
}


TEST_F(UNAR_SCALR, Clip)
{
	check_unar_scalr(parse_line("clip"), [](nnet::varptr in, std::vector<double> params)
	{
		return nnet::clip(in, params[0], params[1]);
	});
}


TEST_F(UNAR_SCALR, ClipNorm)
{
	check_unar_scalr(parse_line("clip_norm"), [](nnet::varptr in, std::vector<double> params)
	{
		return nnet::clip_norm(in, params[0]);
	});
}


TEST_F(UNAR_SCALR, Pow)
{
	check_unar_scalr(parse_line("pow"), [](nnet::varptr in, std::vector<double> params)
	{
		return nnet::pow(in, params[0]);
	});
}


TEST_F(UNAR_SCALR, AddC)
{
	check_unar_scalr(parse_line("add_c"), [](nnet::varptr in, std::vector<double> params)
	{
		return in + params[0];
	});
}


TEST_F(UNAR_SCALR, SubC)
{
	check_unar_scalr(parse_line("sub_c"), [](nnet::varptr in, std::vector<double> params)
	{
		return in - params[0];
	});
}


TEST_F(UNAR_SCALR, CSub)
{
	check_unar_scalr(parse_line("c_sub"), [](nnet::varptr in, std::vector<double> params)
	{
		return params[0] - in;
	});
}


TEST_F(UNAR_SCALR, MulC)
{
	check_unar_scalr(parse_line("mul_c"), [](nnet::varptr in, std::vector<double> params)
	{
		return in * params[0];
	});
}


TEST_F(UNAR_SCALR, DivC)
{
	check_unar_scalr(parse_line("div_c"), [](nnet::varptr in, std::vector<double> params)
	{
		return in / params[0];
	});
}


TEST_F(UNAR_SCALR, CDiv) // Precision problem
{
	check_unar_scalr(parse_line("c_div"), [](nnet::varptr in, std::vector<double> params)
	{
		return params[0] / in;
	});
}


#endif /* DISABLE_TF_UNAR_SCALR_TESTS */
