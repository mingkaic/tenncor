#ifndef DISABLE_TF_TRANS_TESTS

#include "tests/err_tests/include/tf_verify.hpp"

#include "include/graph/leaf/variable.hpp"
#include "include/operations/operations.hpp"


class TRANS : public TF_VERIFY
{
public:
	TRANS (void)
	{
		TF_VERIFY::to_mem("TRANS");
	}
};


void check_trans (op_args args, std::function<nnet::varptr(nnet::varptr,std::vector<double>)> op)
{
	nnet::variable* input = args.vars_[0];
	nnet::variable* expectout = args.vars_[1];
	nnet::variable* expectgrad = args.vars_[2];

	input->initialize();
	expectout->initialize();
	expectgrad->initialize();
	nnet::varptr out = op(nnet::varptr(input), args.params_);

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


TEST_F(TRANS, Transpose)
{
	check_trans(parse_line("transpose"), [](nnet::varptr in, std::vector<double> params)
	{
		return nnet::transpose(in);
	});
}


TEST_F(TRANS, ReduceMax)
{
	check_trans(parse_line("reduce_max_i"), [](nnet::varptr in, std::vector<double> params)
	{
		size_t rank = in->get_tensor()->get_shape().rank();
		return nnet::reduce_max(in, rank - params[0] - 1);
	});
}


// TEST_F(TRANS, ReduceMin)
// {
// 	check_trans(parse_line("reduce_min_i"), [](nnet::varptr in, std::vector<double> params)
// 	{
// 		size_t rank = in->get_tensor()->get_shape().rank();
// 		return nnet::reduce_min(in, rank - params[0] - 1);
// 	});
// }


TEST_F(TRANS, ReduceSum)
{
	check_trans(parse_line("reduce_sum_i"), [](nnet::varptr in, std::vector<double> params)
	{
		size_t rank = in->get_tensor()->get_shape().rank();
		return nnet::reduce_sum(in, rank - params[0] - 1);
	});
}


TEST_F(TRANS, ReduceMean)
{
	check_trans(parse_line("reduce_mean_i"), [](nnet::varptr in, std::vector<double> params)
	{
		size_t rank = in->get_tensor()->get_shape().rank();
		return nnet::reduce_mean(in, rank - params[0] - 1);
	});
}


#endif /* DISABLE_TF_TRANS_TESTS */
