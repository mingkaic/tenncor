#ifndef DISABLE_TF_MATMUL_TESTS

#include "tests/regress/include/tf_verify.hpp"

#include "graph/variable.hpp"
#include "operate/operations.hpp"


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
		op_args args = parse_line(opname);
		nnet::variable* input1 = args.vars_[0];
		nnet::variable* input2 = args.vars_[1];
		nnet::variable* expectout = args.vars_[2];
		nnet::variable* expectgrad1 = args.vars_[3];
		nnet::variable* expectgrad2 = args.vars_[4];

		nnet::varptr out = nnet::matmul(nnet::varptr(input1), nnet::varptr(input2));
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

}


#endif /* DISABLE_TF_MATMUL_TESTS */
