
#ifndef DISABLE_SESSION_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "eteq/eteq.hpp"


TEST(SESSION, Update)
{
	teq::Shape shape;

	eteq::VarptrT<double> a = eteq::make_variable<double>(
		std::vector<double>(shape.n_elems(), 1).data(), shape);
	eteq::NodeptrT<double> b = eteq::make_variable<double>(
		std::vector<double>(shape.n_elems(), 1).data(), shape);
	eteq::NodeptrT<double> c = eteq::make_variable<double>(
		std::vector<double>(shape.n_elems(), 2).data(), shape);

	auto x = convert_to_node(a) + b;
	auto target = x * c;

	// * (target) = undefined
	// `-- + (x) = undefined
	// |   `-- a = 1
	// |   `-- b = 1
	// `-- c = 2

	eteq::Session session;
	session.track({
		target->get_tensor(),
	});
	session.update();

	// expected state:
	// * (target) = 4
	// `-- + (x) = 2
	// |   `-- a = 1
	// |   `-- b = 1
	// `-- c = 2

	double* data = (double*) target->data();
	double* datax = (double*) x->data();
	EXPECT_EQ(4, data[0]);
	EXPECT_EQ(2, datax[0]);

	// since the entire target subtree is tracked,
	// EXPECT: everything is updated
	double d = 2;
	a->assign(&d, shape);
	session.update();

	// expected state:
	// * (target) = 6
	// `-- + (x) = 3
	// |   `-- a = 2
	// |   `-- b = 1
	// `-- c = 2

	data = (double*) target->data();
	datax = (double*) x->data();
	EXPECT_EQ(6, data[0]);
	EXPECT_EQ(3, datax[0]);
}


TEST(SESSION, UpdateIgnore)
{
	teq::Shape shape;

	eteq::VarptrT<double> a = eteq::make_variable<double>(
		std::vector<double>(shape.n_elems(), 1).data(), shape);
	eteq::NodeptrT<double> b = eteq::make_variable<double>(
		std::vector<double>(shape.n_elems(), 1).data(), shape);
	eteq::VarptrT<double> c = eteq::make_variable<double>(
		std::vector<double>(shape.n_elems(), 2).data(), shape);
	eteq::VarptrT<double> d = eteq::make_variable<double>(
		std::vector<double>(shape.n_elems(), 2).data(), shape);

	auto x = convert_to_node(a) + b;
	auto y = x * convert_to_node(c);
	auto target = y - convert_to_node(d);

	// - (targetd) = undefined
	// `-- * (y) = undefined
	// |   `-- + (x) = undefined
	// |   |   `-- a = 1
	// |   |   `-- b = 1
	// |   `-- c = 2
	// `-- d = 2

	eteq::Session session;
	session.track({
		target->get_tensor(),
	});
	session.update();

	// expected state:
	// - (targetd) = 2
	// `-- * (y) = 4
	// |   `-- + (x) = 2
	// |   |   `-- a = 1
	// |   |   `-- b = 1
	// |   `-- c = 2
	// `-- d = 2

	double* data = (double*) target->data();
	double* datax = (double*) x->data();
	double* datay = (double*) y->data();
	EXPECT_EQ(2, data[0]);
	EXPECT_EQ(2, datax[0]);
	EXPECT_EQ(4, datay[0]);

	double e = 3;
	a->assign(&e, shape);
	d->assign(&e, shape);
	session.update({y->get_tensor().get()});

	// expected state:
	// - (targetd) = 1
	// `-- * (y) = 4
	// |   `-- + (x) = 2
	// |   |   `-- a = 3
	// |   |   `-- b = 1
	// |   `-- c = 2
	// `-- d = 3

	data = (double*) target->data();
	datax = (double*) x->data();
	datay = (double*) y->data();
	EXPECT_EQ(1, data[0]);
	EXPECT_EQ(2, datax[0]);
	EXPECT_EQ(4, datay[0]);

	double f = 1;
	c->assign(&f, shape);
	session.update({x->get_tensor().get()});

	// expected state:
	// - (targetd) = -1
	// `-- * (y) = 2
	// |   `-- + (x) = 2
	// |   |   `-- a = 3
	// |   |   `-- b = 1
	// |   `-- c = 1
	// `-- d = 3

	data = (double*) target->data();
	datax = (double*) x->data();
	datay = (double*) y->data();
	EXPECT_EQ(-1, data[0]);
	EXPECT_EQ(2, datax[0]);
	EXPECT_EQ(2, datay[0]);
}


TEST(SESSION, TargetedUpdate)
{
	teq::Shape shape;

	eteq::VarptrT<double> a = eteq::make_variable<double>(
		std::vector<double>(shape.n_elems(), 1).data(), shape);
	eteq::NodeptrT<double> b = eteq::make_variable<double>(
		std::vector<double>(shape.n_elems(), 1).data(), shape);
	eteq::NodeptrT<double> c = eteq::make_variable<double>(
		std::vector<double>(shape.n_elems(), 2).data(), shape);

	auto x = convert_to_node(a) + b;
	auto target = x * c;

	// * (target) = undefined
	// `-- + (x) = undefined
	// |   `-- a = 1
	// |   `-- b = 1
	// `-- c = 2

	eteq::Session session;
	session.track({
		target->get_tensor(),
	});
	session.update();

	// expected state:
	// * (target) = 4
	// `-- + (x) = 2
	// |   `-- a = 1
	// |   `-- b = 1
	// `-- c = 2

	double* data = (double*) target->data();
	double* datax = (double*) x->data();
	EXPECT_EQ(4, data[0]);
	EXPECT_EQ(2, datax[0]);

	double d = 2;
	a->assign(&d, shape);
	session.update_target(eteq::TensSetT{x->get_tensor().get()});

	// expected state:
	// * (target) = 4
	// `-- + (x) = 3
	// |   `-- a = 2
	// |   `-- b = 1
	// `-- c = 2

	// expect target to not be updated
	// expect x to be updated
	data = (double*) target->data();
	datax = (double*) x->data();
	EXPECT_EQ(4, data[0]);
	EXPECT_EQ(3, datax[0]);
}


TEST(SESSION, TargetedUpdateIgnore)
{
	teq::Shape shape;

	eteq::VarptrT<double> a = eteq::make_variable<double>(
		std::vector<double>(shape.n_elems(), 1).data(), shape);
	eteq::NodeptrT<double> b = eteq::make_variable<double>(
		std::vector<double>(shape.n_elems(), 1).data(), shape);
	eteq::VarptrT<double> c = eteq::make_variable<double>(
		std::vector<double>(shape.n_elems(), 2).data(), shape);
	eteq::VarptrT<double> d = eteq::make_variable<double>(
		std::vector<double>(shape.n_elems(), 2).data(), shape);

	auto x = convert_to_node(a) + b;
	auto y = x * convert_to_node(c);
	auto target = y - convert_to_node(d);

	// - (targetd) = undefined
	// `-- * (y) = undefined
	// |   `-- + (x) = undefined
	// |   |   `-- a = 1
	// |   |   `-- b = 1
	// |   `-- c = 2
	// `-- d = 2

	eteq::Session session;
	session.track({
		target->get_tensor(),
	});
	session.update();

	// expected state:
	// - (targetd) = 2
	// `-- * (y) = 4
	// |   `-- + (x) = 2
	// |   |   `-- a = 1
	// |   |   `-- b = 1
	// |   `-- c = 2
	// `-- d = 2

	double* data = (double*) target->data();
	double* datax = (double*) x->data();
	double* datay = (double*) y->data();
	EXPECT_EQ(2, data[0]);
	EXPECT_EQ(2, datax[0]);
	EXPECT_EQ(4, datay[0]);

	double e = 3;
	a->assign(&e, shape);
	c->assign(&e, shape);
	d->assign(&e, shape);
	session.update_target({y->get_tensor().get()}, {x->get_tensor().get()});

	// expected state:
	// - (targetd) = 2
	// `-- * (y) = 6
	// |   `-- + (x) = 2
	// |   |   `-- a = 3
	// |   |   `-- b = 1
	// |   `-- c = 3
	// `-- d = 3

	data = (double*) target->data();
	datax = (double*) x->data();
	datay = (double*) y->data();
	EXPECT_EQ(2, data[0]);
	EXPECT_EQ(2, datax[0]);
	EXPECT_EQ(6, datay[0]);
}


#endif // DISABLE_SESSION_TEST
