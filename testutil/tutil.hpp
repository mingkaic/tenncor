
#ifndef TEST_TUTIL_HPP
#define TEST_TUTIL_HPP

#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "dbg/print/print.hpp"

#include "internal/teq/ileaf.hpp"
#include "internal/teq/ifunctor.hpp"

namespace tutil
{

std::string compare_graph (std::istream& expectstr, teq::iTensor* root,
	bool showshape = true, LabelsMapT labels = {});

std::string compare_graph (std::istream& expectstr, teq::TensptrT root,
	bool showshape = true, LabelsMapT labels = {});

template <typename T>
void check_tensordata (teq::iTensor* xpc, teq::iTensor* got, const char* fname, size_t lno)
{
	if (nullptr == got || nullptr == xpc)
	{
		FAIL() << "cannot check null tensors " << xpc << " " << got
			<< " @" << fname << ":" << lno;
	}
	teq::Shape eshape = xpc->shape();
	teq::Shape gshape = got->shape();
	ASSERT_TRUE(std::equal(eshape.begin(), eshape.end(), gshape.begin()))
		<< fname << ":" << lno << ": "
		<< "expect list " << fmts::to_string(eshape.begin(), eshape.end())
		<< ", got " << fmts::to_string(gshape.begin(), gshape.end()) << " instead";
	auto exvar = dynamic_cast<teq::iLeaf*>(xpc);
	auto govar = dynamic_cast<teq::iLeaf*>(got);
	auto exfnc = dynamic_cast<teq::iFunctor*>(xpc);
	auto gofnc = dynamic_cast<teq::iFunctor*>(got);
	if (nullptr != exvar && nullptr != govar)
	{
		T* expect = (T*) exvar->device().data();
		T* got = (T*) govar->device().data();
		size_t n = eshape.n_elems();
		ASSERT_TRUE(std::equal(expect, expect + n, got))
			<< fname << ":" << lno << ": "
			<< "expect list " << fmts::to_string(expect, expect + n)
			<< ", got " << fmts::to_string(got, got + n) << " instead";
	}
	else if (nullptr != exfnc && nullptr != gofnc)
	{
		T* expect = (T*) exfnc->device().data();
		T* got = (T*) gofnc->device().data();
		size_t n = eshape.n_elems();
		ASSERT_TRUE(std::equal(expect, expect + n, got))
			<< fname << ":" << lno << ": "
			<< "expect list " << fmts::to_string(expect, expect + n)
			<< ", got " << fmts::to_string(got, got + n) << " instead";
	}
	else
	{
		FAIL() << fname << ":" << lno
			<< ": checking non-opfunc and non-ileaf tensors "
			<< xpc->to_string() << " " << got->to_string();
	}
}

template <typename T>
void check_tensordata_real (teq::iTensor* xpc, teq::iTensor* got, const char* fname, size_t lno)
{
	if (nullptr == got || nullptr == xpc)
	{
		FAIL() << "cannot check null tensors " << xpc << " " << got
			<< " @" << fname << ":" << lno;
	}
	teq::Shape eshape = xpc->shape();
	teq::Shape gshape = got->shape();
	ASSERT_TRUE(std::equal(eshape.begin(), eshape.end(), gshape.begin()))
		<< fname << ":" << lno << ": "
		<< "expect list " << fmts::to_string(eshape.begin(), eshape.end())
		<< ", got " << fmts::to_string(gshape.begin(), gshape.end()) << " instead";
	auto exvar = dynamic_cast<teq::iLeaf*>(xpc);
	auto govar = dynamic_cast<teq::iLeaf*>(got);
	auto exfnc = dynamic_cast<teq::iFunctor*>(xpc);
	auto gofnc = dynamic_cast<teq::iFunctor*>(got);
	if (nullptr != exvar && nullptr != govar)
	{
		T* expect = (T*) exvar->device().data();
		T* got = (T*) govar->device().data();
		size_t n = eshape.n_elems();
		for (size_t i = 0; i < n; ++i)
		{
			ASSERT_DOUBLE_EQ(expect[i], got[i])
				<< fname << ":" << lno << ": "
				<< "expect list " << fmts::to_string(expect, expect + n)
				<< ", got " << fmts::to_string(got, got + n) << " instead";
		}
	}
	else if (nullptr != exfnc && nullptr != gofnc)
	{
		T* expect = (T*) exfnc->device().data();
		T* got = (T*) gofnc->device().data();
		size_t n = eshape.n_elems();
		for (size_t i = 0; i < n; ++i)
		{
			ASSERT_DOUBLE_EQ(expect[i], got[i])
				<< fname << ":" << lno << ": "
				<< "expect list " << fmts::to_string(expect, expect + n)
				<< ", got " << fmts::to_string(got, got + n) << " instead";
		}
	}
	else
	{
		FAIL() << fname << ":" << lno
			<< ": checking non-opfunc and non-ileaf tensors "
			<< xpc->to_string() << " " << got->to_string();
	}
}

#define EXPECT_TENSDATA(EXTENS, GOTENS, DTYPE)\
tutil::check_tensordata<DTYPE>(EXTENS, GOTENS, __FILE__, __LINE__)

#define EXPECT_TENSDATA_REAL(EXTENS, GOTENS, DTYPE)\
tutil::check_tensordata_real<DTYPE>(EXTENS, GOTENS, __FILE__, __LINE__)

#define EXPECT_GRAPHEQ(MSG, ROOT) {\
	std::istringstream ss(MSG);\
	auto compare_str = tutil::compare_graph(ss, ROOT);\
	EXPECT_EQ(0, compare_str.size()) << compare_str;\
}

#define EXPECT_ERR(E, MSG)\
EXPECT_NE(nullptr, E);\
if (nullptr != E)\
{\
	EXPECT_STREQ(MSG, E->to_string().c_str());\
}

#define ASSERT_NOERR(ERR) {\
	std::string err_msg;\
	if (nullptr != ERR)\
	{\
		err_msg = ERR->to_string();\
	}\
	ASSERT_EQ(nullptr, ERR) << err_msg;\
}

}

#endif // TEST_TUTIL_HPP
