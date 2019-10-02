
#ifndef DISABLE_SEQMODEL_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "layr/seqmodel.hpp"
#include "layr/dense.hpp"
#include "layr/rbm.hpp"
#include "layr/conv.hpp"


TEST(SEQMODEL, Copy)
{
	std::string dlabel = "especially_dense";
	std::string rlabel = "kinda_restrictive";
	std::string clabel = "maybe_convoluted";
    std::string label = "serious_series";
    layr::SequentialModel model(label);

    model.push_back(std::make_shared<layr::Dense>(4, 5,
		layr::zero_init<PybindT>(),
		layr::zero_init<PybindT>(),
        dlabel));
    model.push_back(std::make_shared<layr::RBM>(6, 4,
        layr::sigmoid(),
		layr::zero_init<PybindT>(),
		layr::zero_init<PybindT>(),
		rlabel));
    model.push_back(std::make_shared<layr::Conv>(
        std::pair<teq::DimT,teq::DimT>{3, 4}, 6, 7, clabel));

	auto expectc = model.get_contents();

    layr::SequentialModel cpy(model);

	EXPECT_STREQ(label.c_str(), cpy.get_label().c_str());
	EXPECT_EQ(5, cpy.get_ninput());
	EXPECT_EQ(7, cpy.get_noutput());

	auto gotc = cpy.get_contents();

	ASSERT_EQ(expectc.size(), gotc.size());
	ASSERT_EQ(9, gotc.size());

    // dense
	ASSERT_NE(expectc[0], gotc[0]);
	ASSERT_NE(expectc[1], gotc[1]);
	EXPECT_STREQ("weight", gotc[0]->to_string().c_str());
	EXPECT_STREQ("bias", gotc[1]->to_string().c_str());
	EXPECT_TENSDATA(expectc[0].get(), gotc[0].get(), PybindT);
	EXPECT_TENSDATA(expectc[1].get(), gotc[1].get(), PybindT);

    // rbm
	ASSERT_NE(expectc[2], gotc[2]);
	ASSERT_NE(expectc[3], gotc[3]);
	ASSERT_NE(expectc[5], gotc[5]);
	EXPECT_STREQ("weight", gotc[2]->to_string().c_str());
	EXPECT_STREQ("bias", gotc[3]->to_string().c_str());
	EXPECT_STREQ("bias", gotc[5]->to_string().c_str());
	EXPECT_TENSDATA(expectc[2].get(), gotc[2].get(), PybindT);
	EXPECT_TENSDATA(expectc[3].get(), gotc[3].get(), PybindT);
	EXPECT_TENSDATA(expectc[5].get(), gotc[5].get(), PybindT);

    // conv
	ASSERT_NE(expectc[7], gotc[7]);
	ASSERT_NE(expectc[8], gotc[8]);
	EXPECT_STREQ("weight", gotc[7]->to_string().c_str());
	EXPECT_STREQ("bias", gotc[8]->to_string().c_str());
	EXPECT_TENSDATA(expectc[7].get(), gotc[7].get(), PybindT);
	EXPECT_TENSDATA(expectc[8].get(), gotc[8].get(), PybindT);
}


#endif // DISABLE_SEQMODEL_TEST
