
#ifndef DISABLE_SESSION_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "ead/ead.hpp"


TEST(SESSION, Update)
{
	ade::Shape shape;

	ead::VarptrT<double> a = ead::make_variable<double>(
		std::vector<double>(shape.n_elems(), 1).data(), shape);
	ead::NodeptrT<double> b = ead::make_variable<double>(
		std::vector<double>(shape.n_elems(), 1).data(), shape);
	ead::NodeptrT<double> c = ead::make_variable<double>(
		std::vector<double>(shape.n_elems(), 2).data(), shape);

	auto x = tenncor::add(convert_to_node(a), b);
	auto target = tenncor::mul(x, c);

	ead::Session session;
	session.track({
		target->get_tensor(),
	});

	session.update();

	double* data = (double*) target->data();
	EXPECT_EQ(4, data[0]);

	double d = 2;
	a->assign(&d, shape);
	session.update();

	// expect target to be updated
	data = (double*) target->data();
	EXPECT_EQ(6, data[0]);
}


TEST(SESSION, TargetedUpdate)
{
	ade::Shape shape;

	ead::VarptrT<double> a = ead::make_variable<double>(
		std::vector<double>(shape.n_elems(), 1).data(), shape);
	ead::NodeptrT<double> b = ead::make_variable<double>(
		std::vector<double>(shape.n_elems(), 1).data(), shape);
	ead::NodeptrT<double> c = ead::make_variable<double>(
		std::vector<double>(shape.n_elems(), 2).data(), shape);

	auto x = tenncor::add(convert_to_node(a), b);
	auto target = tenncor::mul(x, c);

	ead::Session session;
	session.track({
		target->get_tensor(),
	});

	session.update();

	double* data = (double*) target->data();
	EXPECT_EQ(4, data[0]);

	double d = 2;
	a->assign(&d, shape);
	session.update_target(ead::TensSetT{x->get_tensor().get()});

	// expect target to not be updated
	data = (double*) target->data();
	EXPECT_EQ(4, data[0]);

	// expect x to be updated
	data = (double*) x->data();
	EXPECT_EQ(3, data[0]);
}


#endif // DISABLE_SESSION_TEST
