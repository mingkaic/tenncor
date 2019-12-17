#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "dbg/stream/teq.hpp"

#include "teq/ileaf.hpp"
#include "teq/ifunctor.hpp"

#ifndef TEST_TUTIL_HPP
#define TEST_TUTIL_HPP

namespace tutil
{

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
		T* expect = (T*) exvar->data();
		T* got = (T*) govar->data();
		size_t n = eshape.n_elems();
		ASSERT_TRUE(std::equal(expect, expect + n, got))
			<< fname << ":" << lno << ": "
			<< "expect list " << fmts::to_string(expect, expect + n)
			<< ", got " << fmts::to_string(got, got + n) << " instead";
	}
	else if (nullptr != exfnc && nullptr != gofnc)
	{
		T* expect = (T*) exfnc->data();
		T* got = (T*) gofnc->data();
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
		T* expect = (T*) exvar->data();
		T* got = (T*) govar->data();
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
		T* expect = (T*) exfnc->data();
		T* got = (T*) gofnc->data();
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

}

#endif // TEST_TUTIL_HPP
