
#ifndef DISABLE_TEQ_EVALUATOR_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "internal/teq/mock/mock.hpp"

#include "internal/teq/evaluator.hpp"


using ::testing::_;
using ::testing::Invoke;
using ::testing::Return;
using ::testing::ReturnRef;


TEST(EVALUATOR, SetGet)
{
	global::CfgMapptrT ctx = std::make_shared<estd::ConfigMap<>>();

	auto& eval = teq::get_eval(ctx);
	void* origptr = &eval;
	auto oeval = new teq::Evaluator();
	teq::set_eval(oeval, ctx);
	EXPECT_NE(origptr, &teq::get_eval(ctx));
	EXPECT_EQ(oeval, &teq::get_eval(ctx));
}


TEST(EVALUATOR, Update)
{
	teq::Shape shape;

	auto a = make_var(shape);
	auto b = make_var(shape);
	auto c = make_var(shape);

	auto x = make_fnc("", 0, teq::TensptrsT{a, b});
	auto target = make_fnc("", 0, teq::TensptrsT{x, c});

	// before
	// (target) = not updated
	// `-- (x) = not updated
	// |   `-- a
	// |   `-- b
	// `-- c

	// after
	// (target) = updated
	// `-- (x) = updated
	// |   `-- a
	// |   `-- b
	// `-- c
	const teq::iTensor* capx = nullptr;
	const teq::iTensor* captarg = nullptr;
	auto capture_x = [&](const teq::iTensor& arg, size_t){ capx = &arg; };
	auto capture_target = [&](const teq::iTensor& arg, size_t){ captarg = &arg; };

	MockDevice mdevice;
	EXPECT_CALL(mdevice, calc(_,_)).Times(2).
		WillOnce(Invoke(capture_x)).
		WillOnce(Invoke(capture_target));

	teq::Evaluator eval;
	eval.evaluate(mdevice, {target.get()});

	EXPECT_EQ(x.get(), capx);
	EXPECT_EQ(target.get(), captarg);
}


TEST(EVALUATOR, UpdateIgnore)
{
	teq::Shape shape;

	auto a = make_var(shape);
	auto b = make_var(shape);
	auto c = make_var(shape);
	auto d = make_var(shape);

	auto x = make_fnc("", 0, teq::TensptrsT{a, b});
	auto y = make_fnc("", 0, teq::TensptrsT{x, c});
	auto target = make_fnc("", 0, teq::TensptrsT{y, d});

	double mockdata = 0;
	MockDeviceRef devref;
	EXPECT_CALL(*y, device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(*x, device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(devref, data()).WillRepeatedly(Return(&mockdata));

	// before
	// (target) = not updated
	// `-- (y) = not updated
	// |   `-- (x) = not updated
	// |   |   `-- a
	// |   |   `-- b
	// |   `-- c
	// `-- d

	// after
	// (target) = updated
	// `-- (y) = not updated
	// |   `-- (x) = not updated
	// |   |   `-- a
	// |   |   `-- b
	// |   `-- c
	// `-- d
	const teq::iTensor* captarg = nullptr;
	auto capture_target = [&](const teq::iTensor& arg,size_t){ captarg = &arg; };

	MockDevice mdevice;
	EXPECT_CALL(mdevice, calc(_,_)).Times(1).
		WillOnce(Invoke(capture_target));

	teq::Evaluator eval;
	eval.evaluate(mdevice, {target.get()}, {y.get()});

	EXPECT_EQ(target.get(), captarg);

	// after
	// (target) = updated
	// `-- (y) = updated
	// |   `-- (x) = not updated
	// |   |   `-- a
	// |   |   `-- b
	// |   `-- c
	// `-- d
	const teq::iTensor* capy = nullptr;
	const teq::iTensor* captarg2 = nullptr;
	auto capture_y = [&](const teq::iTensor& arg, size_t){ capy = &arg; };
	auto capture_target2 = [&](const teq::iTensor& arg, size_t){ captarg2 = &arg; };

	EXPECT_CALL(mdevice, calc(_,_)).Times(2).
		WillOnce(Invoke(capture_y)).
		WillOnce(Invoke(capture_target2));

	eval.evaluate(mdevice, {target.get()}, {x.get()});

	EXPECT_EQ(y.get(), capy);
	EXPECT_EQ(target.get(), captarg2);
}


TEST(EVALUATOR, UpdateIgnoreCommonDesc)
{
	teq::Shape shape;

	auto a = make_var(shape);
	auto b = make_var(shape);
	auto c = make_var(shape);

	auto u = make_fnc("", 0, teq::TensptrsT{a});
	auto x = make_fnc("", 0, teq::TensptrsT{u, b});
	auto y = make_fnc("", 0, teq::TensptrsT{c, u});
	auto target = make_fnc("", 0, teq::TensptrsT{y, x});

	double mockdata = 0;
	MockDeviceRef devref;
	EXPECT_CALL(*y, device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(*x, device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(*u, device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(devref, data()).WillRepeatedly(Return(&mockdata));

	// before
	// (target) = not updated
	// `-- (y) = not updated
	// |   `-- c
	// |   `-- (u) = not updated
	// |       `-- a
	// `-- (x) = not updated
	//     `-- (u)
	//     |   `-- a
	//     `-- b

	// after
	// (target) = updated
	// `-- (y) = not updated
	// |   `-- c
	// |   `-- (u) = updated
	// |       `-- a
	// `-- (x) = updated
	//     `-- (u)
	//     |   `-- a
	//     `-- b
	const teq::iTensor* capu = nullptr;
	const teq::iTensor* capx = nullptr;
	const teq::iTensor* captarg = nullptr;
	auto capture_u = [&](const teq::iTensor& arg, size_t){ capu = &arg; };
	auto capture_x = [&](const teq::iTensor& arg, size_t){ capx = &arg; };
	auto capture_target = [&](const teq::iTensor& arg, size_t){ captarg = &arg; };

	MockDevice mdevice;
	EXPECT_CALL(mdevice, calc(_,_)).Times(3).
		WillOnce(Invoke(capture_u)).
		WillOnce(Invoke(capture_x)).
		WillOnce(Invoke(capture_target));

	teq::Evaluator eval;
	eval.evaluate(mdevice, {target.get()}, {y.get()});

	EXPECT_EQ(u.get(), capu);
	EXPECT_EQ(x.get(), capx);
	EXPECT_EQ(target.get(), captarg);
}


TEST(EVALUATOR, TargetedUpdate)
{
	teq::Shape shape;

	auto a = make_var(shape);
	auto b = make_var(shape);
	auto c = make_var(shape);

	auto x = make_fnc("", 0, teq::TensptrsT{a, b});
	auto target = make_fnc("", 0, teq::TensptrsT{x, c});

	// before
	// (target) = not updated
	// `-- (x) = not updated
	// |   `-- a
	// |   `-- b
	// `-- c

	// after
	// (target) = not updated
	// `-- (x) = updated
	// |   `-- a
	// |   `-- b
	// `-- c
	const teq::iTensor* capx = nullptr;
	auto capture_x = [&](const teq::iTensor& arg, size_t){ capx = &arg; };

	MockDevice mdevice;
	EXPECT_CALL(mdevice, calc(_,_)).Times(1).
		WillOnce(Invoke(capture_x));

	teq::Evaluator eval;
	eval.evaluate(mdevice, teq::TensSetT{x.get()});

	EXPECT_EQ(x.get(), capx);
}


TEST(EVALUATOR, TargetedUpdateIgnore)
{
	teq::Shape shape;

	auto a = make_var(shape);
	auto b = make_var(shape);
	auto c = make_var(shape);
	auto d = make_var(shape);

	auto x = make_fnc("", 0, teq::TensptrsT{a, b});
	auto y = make_fnc("", 0, teq::TensptrsT{x, c});
	auto target = make_fnc("", 0, teq::TensptrsT{y, d});

	double mockdata = 0;
	MockDeviceRef devref;
	EXPECT_CALL(*y, device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(*x, device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(devref, data()).WillRepeatedly(Return(&mockdata));

	// before
	// (targetd) = not updated
	// `-- (y) = not updated
	// |   `-- (x) = not updated
	// |   |   `-- a
	// |   |   `-- b
	// |   `-- c
	// `-- d

	// after
	// (targetd) = not updated
	// `-- (y) = updated
	// |   `-- (x) = not updated
	// |   |   `-- a
	// |   |   `-- b
	// |   `-- c
	// `-- d
	const teq::iTensor* capy = nullptr;
	auto capture_y = [&](const teq::iTensor& arg, size_t){ capy = &arg; };

	MockDevice mdevice;
	EXPECT_CALL(mdevice, calc(_,_)).Times(1).
		WillOnce(Invoke(capture_y));

	teq::Evaluator eval;
	eval.evaluate(mdevice, {y.get()}, {x.get()});

	EXPECT_EQ(y.get(), capy);
}


TEST(EVALUATOR, TargetedUpdateIgnoreCommonDesc)
{
	teq::Shape shape;

	auto a = make_var(shape);
	auto b = make_var(shape);
	auto c = make_var(shape);
	auto d = make_var(shape);

	auto u = make_fnc("", 0, teq::TensptrsT{a});
	auto x = make_fnc("", 0, teq::TensptrsT{u, b});
	auto y = make_fnc("", 0, teq::TensptrsT{c, u});
	auto z = make_fnc("", 0, teq::TensptrsT{y, x});
	auto target = make_fnc("", 0, teq::TensptrsT{z, d});

	double mockdata = 0;
	MockDeviceRef devref;
	EXPECT_CALL(*z, device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(*y, device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(*x, device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(*u, device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(devref, data()).WillRepeatedly(Return(&mockdata));

	// before
	// (targeted) = not updated
	// `-- (z) = not updated
	// |   `-- (y) = not updated
	// |   |   `-- c
	// |   |   `-- (u) = not updated
	// |   |       `-- a
	// |   `-- (x) = not updated
	// |       `-- (u)
	// |       |   `-- a
	// |       `-- b
	// `-- d

	// after
	// (targeted) = not updated
	// `-- (z) = updated
	// |   `-- (y) = not updated
	// |   |   `-- c
	// |   |   `-- (u) = updated
	// |   |       `-- a
	// |   `-- (x) = updated
	// |       `-- (u)
	// |       |   `-- a
	// |       `-- b
	// `-- d
	const teq::iTensor* capu = nullptr;
	const teq::iTensor* capx = nullptr;
	const teq::iTensor* capz = nullptr;
	auto capture_u = [&](const teq::iTensor& arg, size_t){ capu = &arg; };
	auto capture_x = [&](const teq::iTensor& arg, size_t){ capx = &arg; };
	auto capture_z = [&](const teq::iTensor& arg, size_t){ capz = &arg; };

	MockDevice mdevice;
	EXPECT_CALL(mdevice, calc(_,_)).Times(3).
		WillOnce(Invoke(capture_u)).
		WillOnce(Invoke(capture_x)).
		WillOnce(Invoke(capture_z));

	teq::Evaluator eval;
	eval.evaluate(mdevice, {z.get()}, {y.get()});

	EXPECT_EQ(u.get(), capu);
	EXPECT_EQ(x.get(), capx);
	EXPECT_EQ(z.get(), capz);
}


#endif // DISABLE_TEQ_EVALUATOR_TEST
