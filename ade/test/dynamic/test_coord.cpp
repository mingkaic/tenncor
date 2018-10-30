
#ifndef DISABLE_COORD_TEST


#include "gtest/gtest.h"

#include "ade/coord.hpp"

#include "testutil/common.hpp"


struct COORD : public simple::TestModel {};


TEST_F(COORD, String)
{
}


TEST_F(COORD, REchelon)
{
}


TEST_F(COORD, Inverse)
{
	simple::SessionT sess = get_session("COORD::Inverse");

    ade::MatrixT out, in;
    std::vector<double> indata = sess->get_double("indata", ade::mat_dim * ade::mat_dim, {0.0001, 5});
    for (uint8_t i = 0; i < ade::mat_dim; ++i)
    {
        for (uint8_t j = 0; j < ade::mat_dim; ++j)
        {
            in[i][j] = indata[i * ade::mat_dim + j];
        }
    }

    ade::inverse(out, in);

    // expect matmul is identity
    for (uint8_t i = 0; i < ade::mat_dim; ++i)
    {
        for (uint8_t j = 0; j < ade::mat_dim; ++j)
        {
            double val = 0;
            for (uint8_t k = 0; k < ade::mat_dim; ++k)
            {
                val += out[i][k] * in[k][j];
            }
            if (i == j)
            {
                EXPECT_DOUBLE_EQ(1, std::round(val));
            }
            else
            {
                EXPECT_DOUBLE_EQ(0, std::round(val));
            }
        }
    }
}


#endif // DISABLE_COORD_TEST
