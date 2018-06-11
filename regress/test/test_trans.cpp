#ifndef DISABLE_TF_TRANS_TESTS

#include "regressutil/tf_verify.hpp"

#include "kiln/variable.hpp"
#include "kiln/operators.hpp"
#include "kiln/delta.hpp"


class TRANS_TESTS : public TF_VERIFY
{
public:
	TRANS_TESTS (void)
	{
		TF_VERIFY::to_mem("TRANS");
	}
};


using UNARY_OP = std::function<kiln::Identifier*(kiln::Variable*,std::vector<double>)>;


void check_trans (OpArgs args, UNARY_OP op)
{
	assert(args.vars_.size() == 3);
	kiln::Variable* input = args.vars_[0];
	kiln::Variable* expectout = args.vars_[1];
	kiln::Variable* expectgrad = args.vars_[2];

	kiln::Graph::get_global().initialize_all();
	kiln::Identifier* out = op(input, args.params_);

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


TEST_F(TRANS_TESTS, Transpose)
{
	check_trans(parse_line("transpose"),
	[](kiln::Variable* in, std::vector<double> params)
	{
		return kiln::transpose(in);
	});
}


TEST_F(TRANS_TESTS, ReduceMax)
{
	check_trans(parse_line("reduce_max_i"),
	[](kiln::Variable* in, std::vector<double> params)
	{
		size_t rank = in->get()->get_shape().rank();
		return kiln::reduce_max(in, rank - params[0] - 1);
	});
}


TEST_F(TRANS_TESTS, ReduceSum)
{
	check_trans(parse_line("reduce_sum_i"),
	[](kiln::Variable* in, std::vector<double> params)
	{
		size_t rank = in->get()->get_shape().rank();
		return kiln::reduce_sum(in, rank - params[0] - 1);
	});
}


TEST_F(TRANS_TESTS, ReduceMean)
{
	check_trans(parse_line("reduce_mean_i"),
	[](kiln::Variable* in, std::vector<double> params)
	{
		size_t rank = in->get()->get_shape().rank();
		return kiln::reduce_mean(in, rank - params[0] - 1);
	});
}


#endif /* DISABLE_TF_TRANS_TESTS */
