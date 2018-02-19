//
// Created by Mingkai Chen on 2016-12-27.
//

#ifndef DISABLE_CONNECTOR_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"

#include "tests/include/utils/fuzz.h"


#define DISABLE_MUTABLE_TEST
#ifndef DISABLE_MUTABLE_TEST


class MUTABLE : public FUZZ::fuzz_test {};


TEST_F(MUTABLE, connector)
{
	// mutables avoid killing constants by killing dependencies
	// instead of safely destroying its permanent connector node
	nnet::mutable_connector<double>* temp =
		nnet::mutable_connector<double>::build(
		[](std::vector<nnet::varptr >& args) -> nnet::inode*
		{
			return args[0] + args[1];
		}, 2);

	nnet::constant* s1 = nnet::constant::build(10);
	nnet::constant* s2 = nnet::constant::build(20);
	temp->add_arg(s1, 1);
	temp->add_arg(s1, 0);
	// temp is now 20
	double res = nnet::expose<double>(temp)[0];
	EXPECT_EQ(20, res);

	temp->add_arg(s2, 1);
	// temp is now 30
	res = nnet::expose<double>(temp)[0];
	EXPECT_EQ(30, res);

	temp->remove_arg(0);
	// temp is now invalid
	EXPECT_EQ(nullptr, temp->get_eval());

	delete s1;
	delete s2;
	delete temp;
}


TEST_F(MUTABLE, deletion)
{
	nnet::mutable_connector<double>* temp =
		nnet::mutable_connector<double>::build(
		[](std::vector<nnet::varptr >& args) -> nnet::inode*
		{
			return args[0] + args[1];
		}, 2);

	nnet::variable* s1 = new nnet::variable(10);
	nnet::variable* s2 = new nnet::variable(20);
	temp->add_arg(s1, 1);
	temp->add_arg(s1, 0);

	delete s1;
	delete s2;
}


#endif /* DISABLE_MUTABLE_TEST */



#endif /* DISABLE_CONNECTOR_MODULE_TESTS */

