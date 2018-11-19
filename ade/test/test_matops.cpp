
#ifndef DISABLE_MATOPS_TEST


#include "gtest/gtest.h"

#include "ade/matops.hpp"

#include "testutil/common.hpp"


struct MATOPS : public simple::TestModel
{
	virtual void TearDown (void)
	{
		simple::TestModel::TearDown();
		TestLogger::latest_warning_ = "";
		TestLogger::latest_error_ = "";
	}
};


TEST_F(MATOPS, ToString)
{
	std::string expected = "[[0\\1\\2\\3\\4\\5\\6\\7\\8]\\\n"
		"[9\\10\\11\\12\\13\\14\\15\\16\\17]\\\n"
		"[18\\19\\20\\21\\22\\23\\24\\25\\26]\\\n"
		"[27\\28\\29\\30\\31\\32\\33\\34\\35]\\\n"
		"[36\\37\\38\\39\\40\\41\\42\\43\\44]\\\n"
		"[45\\46\\47\\48\\49\\50\\51\\52\\53]\\\n"
		"[54\\55\\56\\57\\58\\59\\60\\61\\62]\\\n"
		"[63\\64\\65\\66\\67\\68\\69\\70\\71]\\\n"
		"[72\\73\\74\\75\\76\\77\\78\\79\\80]]";
	ade::MatrixT mat;
	for (uint8_t i = 0; i < ade::mat_dim; ++i)
	{
		for (uint8_t j = 0; j < ade::mat_dim; ++j)
		{
			mat[i][j] = i * ade::mat_dim + j;
		}
	}
	EXPECT_STREQ(expected.c_str(), ade::to_string(mat).c_str());
}


TEST_F(MATOPS, Inverse)
{
	simple::SessionT sess = get_session("MATOPS::Inverse");

	ade::MatrixT out, in;
	ade::MatrixT zout, zin;
	ade::MatrixT badout, badin;
	std::vector<double> indata = sess->get_double("indata",
		ade::mat_dim * ade::mat_dim, {0.0001, 5});
	for (uint8_t i = 0; i < ade::mat_dim; ++i)
	{
		for (uint8_t j = 0; j < ade::mat_dim; ++j)
		{
			in[i][j] = indata[i * ade::mat_dim + j];
		}
		int zidx = 0;
		if (i > 3)
		{
			zidx = sess->get_int(err::sprintf("zidx%d", i), 1, {1, i - 1})[0];
		}
		for (uint8_t j = 0; j < zidx; ++j)
		{
			zin[i][j] = 0;
		}
		for (uint8_t j = zidx; j < ade::mat_dim; ++j)
		{
			zin[i][j] = indata[i * ade::mat_dim + j];
		}
	}
	std::memset(badin, 0, ade::mat_size);
	for (uint8_t j = 0; j < ade::mat_dim; ++j)
	{
		badin[0][j] = indata[j];
	}

	ade::inverse(out, in);
	ade::inverse(zout, zin);

	std::string fatalmsg = err::sprintf("cannot invert matrix:\n%s",
		ade::to_string(badin).c_str());
	EXPECT_FATAL(ade::inverse(badout, badin), fatalmsg.c_str());

	// expect matmul is identity
	for (uint8_t i = 0; i < ade::mat_dim; ++i)
	{
		for (uint8_t j = 0; j < ade::mat_dim; ++j)
		{
			double val = 0;
			double zval = 0;
			for (uint8_t k = 0; k < ade::mat_dim; ++k)
			{
				val += out[i][k] * in[k][j];
				zval += zout[i][k] * zin[k][j];
			}
			if (i == j)
			{
				EXPECT_DOUBLE_EQ(1, std::round(val));
				EXPECT_DOUBLE_EQ(1, std::round(zval));
			}
			else
			{
				EXPECT_DOUBLE_EQ(0, std::round(val));
				EXPECT_DOUBLE_EQ(0, std::round(zval));
			}
		}
	}
}


TEST_F(MATOPS, Matmul)
{
	simple::SessionT sess = get_session("MATOPS::Matmul");

	ade::MatrixT expected, out, in, in2;
	std::vector<double> indata = sess->get_double("indata",
		ade::mat_dim * ade::mat_dim, {0.0001, 5});
	std::vector<double> indata2 = sess->get_double("indata2",
		ade::mat_dim * ade::mat_dim, {0.0001, 5});
	for (uint8_t i = 0; i < ade::mat_dim; ++i)
	{
		for (uint8_t j = 0; j < ade::mat_dim; ++j)
		{
			in[i][j] = indata[i * ade::mat_dim + j];
			in2[i][j] = indata2[i * ade::mat_dim + j];
			expected[i][j] = 0;
			for (uint8_t k = 0; k < ade::mat_dim; ++k)
			{
				expected[i][j] += indata[i * ade::mat_dim + k] * indata2[k * ade::mat_dim + j];
			}
		}
	}

	ade::matmul(out, in, in2);

	for (uint8_t i = 0; i < ade::mat_dim; ++i)
	{
		for (uint8_t j = 0; j < ade::mat_dim; ++j)
		{
			EXPECT_EQ(expected[i][j], out[i][j]);
		}
	}
}


#endif // DISABLE_MATOPS_TEST
