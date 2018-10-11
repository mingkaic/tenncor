#include "ade/log.hpp"
#include "ade/string.hpp"
#include "ade/shape.hpp"
#include "ade/functor.hpp"

#include "simple/jack.hpp"

struct TestLogger : public ade::iLogger
{
	static std::string latest_warning_;
	static std::string latest_error_;
	static std::string latest_fatal_;

	void warn (std::string msg) const override
	{
		latest_warning_ = msg;
	}

	void error (std::string msg) const override
	{
		latest_error_ = msg;
	}

	void fatal (std::string msg) const override
	{
		latest_fatal_ = msg;
		throw std::runtime_error(latest_fatal_);
	}
};

const std::shared_ptr<TestLogger> tlogger = std::make_shared<TestLogger>();

const size_t nelem_limit = 32456;

#define ASSERT_ARREQ(ARR, ARR2) {\
	std::stringstream arrs, arrs2;\
	ade::to_stream(arrs, ARR);\
	ade::to_stream(arrs2, ARR2);\
	ASSERT_TRUE(std::equal(ARR.begin(), ARR.end(), ARR2.begin())) <<\
		"expect list " << arrs.str() << ", got " << arrs2.str() << " instead"; }

#define EXPECT_ARREQ(ARR, ARR2) {\
	std::stringstream arrs, arrs2;\
	ade::to_stream(arrs, ARR);\
	ade::to_stream(arrs2, ARR2);\
	EXPECT_TRUE(std::equal(ARR.begin(), ARR.end(), ARR2.begin())) <<\
		"expect list " << arrs.str() << ", got " << arrs2.str() << " instead"; }

std::vector<ade::DimT> get_shape_n (SESSION& sess, size_t n, std::string label);

std::vector<ade::DimT> get_shape (SESSION& sess, std::string label);

std::vector<ade::DimT> get_zeroshape (SESSION& sess, std::string label);

std::vector<ade::DimT> get_longshape (SESSION& sess, std::string label);

std::vector<ade::DimT> get_incompatible (SESSION& sess,
	std::vector<ade::DimT> inshape, std::string label);

void int_verify (SESSION& sess, std::string key,
	std::vector<int32_t> data, std::function<void()> verify);

void double_verify (SESSION& sess, std::string key,
	std::vector<double> data, std::function<void()> verify);
