#ifndef DISABLE_TF_TRANS_TESTS

#include "regressutil/tf_verify.hpp"

#include "wire/variable.hpp"
#include "wire/operators.hpp"
#include "wire/delta.hpp"


class TRANS_TESTS : public TF_VERIFY
{
public:
	TRANS_TESTS (void)
	{
		TF_VERIFY::to_mem("TRANS");
	}
};


using UNARY_OP = std::function<wire::Identifier*(wire::Variable*,std::vector<double>)>;


void check_trans (OpArgs args, UNARY_OP op)
{
	assert(args.vars_.size() == 3);
	wire::Variable* input = args.vars_[0];
	wire::Variable* expectout = args.vars_[1];
	wire::Variable* expectgrad = args.vars_[2];

	wire::Graph::get_global().initialize_all();
	wire::Identifier* out = op(input, args.params_);

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


TEST_F(TRANS_TESTS, Transpose)
{
	check_trans(parse_line("transpose"),
	[](wire::Variable* in, std::vector<double> params)
	{
		return wire::transpose(in);
	});
}


TEST_F(TRANS_TESTS, ReduceMax)
{
	check_trans(parse_line("reduce_max_i"),
	[](wire::Variable* in, std::vector<double> params)
	{
		size_t rank = in->get_state().shape_.rank();
		return wire::reduce_max(in, rank - params[0] - 1);
	});
}


TEST_F(TRANS_TESTS, ReduceSum)
{
	check_trans(parse_line("reduce_sum_i"),
	[](wire::Variable* in, std::vector<double> params)
	{
		size_t rank = in->get_state().shape_.rank();
		return wire::reduce_sum(in, rank - params[0] - 1);
	});
}


TEST_F(TRANS_TESTS, ReduceMean)
{
	check_trans(parse_line("reduce_mean_i"),
	[](wire::Variable* in, std::vector<double> params)
	{
		size_t rank = in->get_state().shape_.rank();
		return wire::reduce_mean(in, rank - params[0] - 1);
	});
}


#endif /* DISABLE_TF_TRANS_TESTS */
