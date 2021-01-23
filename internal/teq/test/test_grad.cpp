
#ifndef DISABLE_TEQ_GRAD_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "internal/teq/mock/mock.hpp"

#include "internal/teq/derive.hpp"


using ::testing::_;
using ::testing::Return;


TEST(GRAD, OneZero)
{
	auto w1n = std::make_shared<MockLeaf>();
	auto zr0 = std::make_shared<MockLeaf>();

	MockDerivativeFunc builder;
	teq::TensptrT trash = std::make_shared<MockLeaf>();
	EXPECT_CALL(builder, lderive(_,_,_)).WillRepeatedly(Return(trash));
	EXPECT_CALL(builder, add(_)).WillRepeatedly(Return(trash));

	teq::DimsT slist = {94, 78, 70, 82, 62, 29, 38};
	teq::Shape shape(slist);

	/* standard v
	 *
	 * leaf  leaf2
	 *   \    /
	 *   FUNC
	 */
	auto leaf = make_var(shape, "leaf");
	auto leaf2 = make_var(shape, "leaf2");
	auto leaf3 = make_var(shape, "leaf3");
	auto f = make_fnc("FUNC", 0, teq::TensptrsT{leaf, leaf2});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

	EXPECT_CALL(builder, get_const_one(_)).Times(3).WillRepeatedly(Return(w1n));
	auto wun = teq::derive(f, {f}, builder)[0];
	auto wun2 = teq::derive(leaf, {leaf}, builder)[0];
	auto wun3 = teq::derive(leaf3, {leaf3}, builder)[0];

	EXPECT_EQ(w1n, wun);
	EXPECT_EQ(w1n, wun2);
	EXPECT_EQ(w1n, wun3);

	EXPECT_CALL(*f, ls_attrs()).WillRepeatedly(Return(types::StringsT{}));
	EXPECT_CALL(builder, get_const_one(_)).Times(4).WillRepeatedly(Return(w1n));
	EXPECT_CALL(builder, get_const_zero(_)).Times(5).WillRepeatedly(Return(zr0));
	auto zro = teq::derive(leaf, {leaf3}, builder)[0];
	auto zro2 = teq::derive(leaf3, {leaf}, builder)[0];
	auto zro3 = teq::derive(f, {leaf3}, builder)[0];
	auto zro4 = teq::derive(leaf, {nullptr}, builder)[0];
	auto zro5 = teq::derive(nullptr, {leaf}, builder)[0];

	EXPECT_EQ(zr0, zro);
	EXPECT_EQ(zr0, zro2);
	EXPECT_EQ(zr0, zro3);
	EXPECT_EQ(zr0, zro4);
	EXPECT_EQ(zr0, zro5);
}


TEST(GRAD, BuilderStandardV)
{
	teq::TensptrT w1n = std::make_shared<MockLeaf>();
	teq::TensptrT gexpect = std::make_shared<MockLeaf>();
	teq::TensptrT gexpect2 = std::make_shared<MockLeaf>();

	MockDerivativeFunc builder;
	teq::TensptrT trash = std::make_shared<MockLeaf>();
	EXPECT_CALL(builder, lderive(_,_,_)).WillRepeatedly(Return(trash));
	EXPECT_CALL(builder, get_const_one(_)).WillRepeatedly(Return(trash));
	EXPECT_CALL(builder, get_const_zero(_)).WillRepeatedly(Return(trash));
	EXPECT_CALL(builder, add(_)).WillRepeatedly(Return(trash));

	teq::DimsT slist = {94, 78, 70, 82, 62, 29, 38};
	teq::Shape shape(slist);

	/* standard v
	 *
	 * leaf  leaf2
	 *   \    /
	 *   FUNC
	 */
	auto leaf = make_var(shape, "leaf");
	auto leaf2 = make_var(shape, "leaf2");
	auto f = make_fnc("FUNC", 0, teq::TensptrsT{leaf, leaf2});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*f, ls_attrs()).WillRepeatedly(Return(types::StringsT{}));
	EXPECT_CALL(builder, get_const_one(_)).Times(2).WillRepeatedly(Return(w1n));
	EXPECT_CALL(builder, lderive(teq::FuncptrT(f),w1n,0)).Times(1).WillOnce(Return(gexpect));
	EXPECT_CALL(builder, lderive(teq::FuncptrT(f),w1n,1)).Times(1).WillOnce(Return(gexpect2));

	auto gl = teq::derive(f, {leaf}, builder)[0];
	auto gl2 = teq::derive(f, {leaf2}, builder)[0];

	EXPECT_EQ(gexpect, gl);
	EXPECT_EQ(gexpect2, gl2);
}


TEST(GRAD, BuilderDiamond)
{
	teq::TensptrT w1n = std::make_shared<MockLeaf>();
	teq::TensptrT gexpect = std::make_shared<MockLeaf>();
	teq::TensptrT gexpect2 = std::make_shared<MockLeaf>();
	teq::TensptrT gexpect3 = std::make_shared<MockLeaf>();
	teq::TensptrT gexpect4 = std::make_shared<MockLeaf>();
	teq::TensptrT gexpect5 = std::make_shared<MockLeaf>();

	MockDerivativeFunc builder;
	teq::TensptrT trash = std::make_shared<MockLeaf>();
	EXPECT_CALL(builder, lderive(_,_,_)).WillRepeatedly(Return(trash));
	EXPECT_CALL(builder, get_const_one(_)).WillRepeatedly(Return(trash));
	EXPECT_CALL(builder, get_const_zero(_)).WillRepeatedly(Return(trash));
	EXPECT_CALL(builder, add(_)).WillRepeatedly(Return(trash));

	teq::DimsT slist = {94, 78, 70, 82, 62, 29, 38};
	teq::Shape shape(slist);

	/* diamond
	 *
	 *   leaf
	 *   /   \
	 * FUNC  FUNC2
	 *   \   /
	 *   FUNC3
	 */
	auto leaf = make_var(shape, "leaf");
	auto f = make_fnc("FUNC", 0, teq::TensptrsT{leaf});
	auto f2 = make_fnc("FUNC2", 1, teq::TensptrsT{leaf});
	auto f3 = make_fnc("FUNC3", 2, teq::TensptrsT{f, f2});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*f2, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*f3, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*f, ls_attrs()).WillRepeatedly(Return(types::StringsT{}));
	EXPECT_CALL(*f2, ls_attrs()).WillRepeatedly(Return(types::StringsT{}));
	EXPECT_CALL(*f3, ls_attrs()).WillRepeatedly(Return(types::StringsT{}));
	EXPECT_CALL(builder, get_const_one(_)).Times(1).WillRepeatedly(Return(w1n));
	EXPECT_CALL(builder, lderive(teq::FuncptrT(f3),w1n,0)).Times(1).WillOnce(Return(gexpect));
	EXPECT_CALL(builder, lderive(teq::FuncptrT(f3),w1n,1)).Times(1).WillOnce(Return(gexpect2));
	EXPECT_CALL(builder, lderive(teq::FuncptrT(f),gexpect,0)).Times(1).WillOnce(Return(gexpect3));
	EXPECT_CALL(builder, lderive(teq::FuncptrT(f2),gexpect2,0)).Times(1).WillOnce(Return(gexpect4));
	EXPECT_CALL(builder, add(teq::TensptrsT{gexpect4,gexpect3})).Times(1).WillOnce(Return(gexpect5));

	auto gl = teq::derive(f3, {leaf}, builder)[0];
	EXPECT_EQ(gexpect5, gl);
}


TEST(GRAD, SymmetricalDiamond)
{
	teq::TensptrT w1n = std::make_shared<MockLeaf>();
	teq::TensptrT gexpect = std::make_shared<MockLeaf>();
	teq::TensptrT gexpect2 = std::make_shared<MockLeaf>();
	teq::TensptrT gexpect3 = std::make_shared<MockLeaf>();
	teq::TensptrT gexpect4 = std::make_shared<MockLeaf>();

	MockDerivativeFunc builder;
	teq::TensptrT trash = std::make_shared<MockLeaf>();
	EXPECT_CALL(builder, lderive(_,_,_)).WillRepeatedly(Return(trash));
	EXPECT_CALL(builder, get_const_one(_)).WillRepeatedly(Return(trash));
	EXPECT_CALL(builder, get_const_zero(_)).WillRepeatedly(Return(trash));
	EXPECT_CALL(builder, add(_)).WillRepeatedly(Return(trash));

	teq::DimsT slist = {94, 78, 70, 82, 62, 29, 38};
	teq::Shape shape(slist);

	/* diamond
	 *
	 *   leaf
	 *   /   \
	 * FUNC  FUNC
	 *   \   /
	 *   FUNC2
	 */
	auto leaf = make_var(shape, "leaf");
	auto f = make_fnc("FUNC", 0, teq::TensptrsT{leaf});
	auto f2 = make_fnc("FUNC2", 1, teq::TensptrsT{f,f});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*f2, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*f, ls_attrs()).WillRepeatedly(Return(types::StringsT{}));
	EXPECT_CALL(*f2, ls_attrs()).WillRepeatedly(Return(types::StringsT{}));
	EXPECT_CALL(builder, get_const_one(_)).Times(1).WillRepeatedly(Return(w1n));
	EXPECT_CALL(builder, lderive(teq::FuncptrT(f2),w1n,0)).Times(1).WillOnce(Return(gexpect));
	EXPECT_CALL(builder, lderive(teq::FuncptrT(f2),w1n,1)).Times(1).WillOnce(Return(gexpect2));
	EXPECT_CALL(builder, add(teq::TensptrsT{gexpect,gexpect2})).Times(1).WillOnce(Return(gexpect3));
	EXPECT_CALL(builder, lderive(teq::FuncptrT(f),gexpect3,0)).Times(1).WillOnce(Return(gexpect4));

	auto gl = teq::derive(f2, {leaf}, builder)[0];
	EXPECT_EQ(gexpect4, gl);
}


TEST(GRAD, TadPole)
{
	teq::TensptrT w1n = std::make_shared<MockLeaf>();
	teq::TensptrT gexpect = std::make_shared<MockLeaf>();
	teq::TensptrT gexpect2 = std::make_shared<MockLeaf>();
	teq::TensptrT gexpect3 = std::make_shared<MockLeaf>();
	teq::TensptrT gexpect4 = std::make_shared<MockLeaf>();
	teq::TensptrT gexpect5 = std::make_shared<MockLeaf>();
	teq::TensptrT gexpect6 = std::make_shared<MockLeaf>();

	MockDerivativeFunc builder;

	teq::DimsT slist = {94, 78, 70, 82, 62, 29, 38};
	teq::Shape shape(slist);

	/* diamond with a tail
	 *
	 *   leaf
	 *    |
	 *   FUNC
	 *   /   \
	 * FUNC2  FUNC3
	 *   \   /
	 *   FUNC4
	 */
	auto leaf = make_var(shape, "leaf");
	auto f = make_fnc("FUNC", 0, teq::TensptrsT{leaf});
	auto f2 = make_fnc("FUNC2", 1, teq::TensptrsT{f});
	auto f3 = make_fnc("FUNC3", 2, teq::TensptrsT{f});
	auto f4 = make_fnc("FUNC4", 3, teq::TensptrsT{f2,f3});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*f2, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*f3, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*f4, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*f, ls_attrs()).WillRepeatedly(Return(types::StringsT{}));
	EXPECT_CALL(*f2, ls_attrs()).WillRepeatedly(Return(types::StringsT{}));
	EXPECT_CALL(*f3, ls_attrs()).WillRepeatedly(Return(types::StringsT{}));
	EXPECT_CALL(*f4, ls_attrs()).WillRepeatedly(Return(types::StringsT{}));
	EXPECT_CALL(builder, get_const_one(_)).Times(1).WillRepeatedly(Return(w1n));
	EXPECT_CALL(builder, lderive(teq::FuncptrT(f4),w1n,0)).Times(1).WillOnce(Return(gexpect));
	EXPECT_CALL(builder, lderive(teq::FuncptrT(f4),w1n,1)).Times(1).WillOnce(Return(gexpect2));
	EXPECT_CALL(builder, lderive(teq::FuncptrT(f2),gexpect,0)).Times(1).WillOnce(Return(gexpect3));
	EXPECT_CALL(builder, lderive(teq::FuncptrT(f3),gexpect2,0)).Times(1).WillOnce(Return(gexpect4));
	EXPECT_CALL(builder, add(teq::TensptrsT{gexpect4,gexpect3})).Times(1).WillOnce(Return(gexpect5));
	EXPECT_CALL(builder, lderive(teq::FuncptrT(f),gexpect5,0)).Times(1).WillOnce(Return(gexpect6));

	auto gl = teq::derive(f4, {leaf}, builder)[0];
	EXPECT_EQ(gexpect6, gl);
}


#endif // DISABLE_TEQ_GRAD_TEST
