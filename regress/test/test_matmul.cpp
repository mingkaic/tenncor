#ifndef DISABLE_TF_MATMUL_TESTS

#include "regressutil/tf_verify.hpp"

#include "kiln/variable.hpp"
#include "kiln/operators.hpp"
#include "kiln/delta.hpp"


class MATMUL_TESTS : public TF_VERIFY
{
public:
	MATMUL_TESTS (void)
	{
		TF_VERIFY::to_mem("MATMUL");
	}
};


TEST_F(MATMUL_TESTS, MATMUL_TESTS)
{
	char opname[24];
	for (int i = 0; i < 10; ++i)
	{
		std::sprintf(opname, "matmul%d", i);
		OpArgs args = parse_line(opname);
		assert(args.vars_.size() == 5);
		kiln::Variable* input1 = args.vars_[0];
		kiln::Variable* input2 = args.vars_[1];
		kiln::Variable* expectout = args.vars_[2];
		kiln::Variable* expectgrad1 = args.vars_[3];
		kiln::Variable* expectgrad2 = args.vars_[4];

		kiln::Identifier* out = kiln::matmul(input1, input2);
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

}


#endif /* DISABLE_TF_MATMUL_TESTS */
