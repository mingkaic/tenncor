#ifndef DISABLE_TF_MATMUL_TESTS

#include "regressutil/tf_verify.hpp"

#include "wire/variable.hpp"
#include "wire/operators.hpp"


class MATMUL_TESTS : public TF_VERIFY
{
public:
	MATMUL_TESTS (void)
	{
		TF_VERIFY::to_mem("MATMUL");
	}
};


TEST_F(MATMUL_TESTS, DISABLED_MATMUL_TESTS)
{
	char opname[24];
	for (int i = 0; i < 10; ++i)
	{
		std::sprintf(opname, "matmul%d", i);
		OpArgs args = parse_line(opname);
		assert(args.vars_.size() == 5);
		wire::Variable* input1 = args.vars_[0];
		wire::Variable* input2 = args.vars_[1];
		wire::Variable* expectout = args.vars_[2];
		wire::Variable* expectgrad1 = args.vars_[3];
		wire::Variable* expectgrad2 = args.vars_[4];

		wire::Identifier* out = wire::matmul(input1, input2);
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

}


#endif /* DISABLE_TF_MATMUL_TESTS */
