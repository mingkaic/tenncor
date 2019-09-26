
#ifndef DISABLE_COORD_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "teq/coord.hpp"


TEST(COORD, Forward)
{
	std::vector<double> indata = {
		0.0019825081, 0.3347071004, 0.0865707708, 0.5146660164, 0.2166590070, 0.5496622507, 0.5109064577, 0.3955951994, 0.3905624328,
		0.9829808762, 0.2214665272, 0.3763727124, 0.3803291189, 0.0148871537, 0.2415570980, 0.9847303063, 0.6160465024, 0.2984319681,
		0.3228383176, 0.1824581171, 0.5475718527, 0.1648524839, 0.0991830687, 0.5106522024, 0.7745806821, 0.7629107131, 0.7852067321,
		0.0340754759, 0.9729239195, 0.8418331382, 0.2592043663, 0.8885644457, 0.7729255083, 0.0479479775, 0.0280240361, 0.0962906387,
		0.6574112228, 0.3108874897, 0.3945660904, 0.1115822221, 0.7803675660, 0.0243814025, 0.8167306564, 0.1627987285, 0.9747075956,
		0.4719070202, 0.1045280700, 0.2537814297, 0.3034878216, 0.6577284931, 0.7127312796, 0.0413897007, 0.1872838523, 0.3953047978,
		0.7792631596, 0.8912496040, 0.6674490687, 0.3525210754, 0.6992469203, 0.8542489297, 0.4675135871, 0.1806591743, 0.7553055254,
		0.0170820545, 0.8560925057, 0.2466748155, 0.6170145637, 0.4385683546, 0.7932879342, 0.8587740827, 0.6226108560, 0.8299188099,
		0.5004368159, 0.0859681124, 0.1584709394, 0.1243801540, 0.4382915214, 0.7866826403, 0.2735058260, 0.6744913352, 0.8209332655,
	};
	std::vector<double> indata2 = {
		0.2529910483, 0.7279482959, 0.6723602388, 0.3115280504, 0.9510976777, 0.3661420357, 0.9339699165, 0.3563378037, 0.8210839592,
		0.9207320041, 0.6338042259, 0.9979807409, 0.7730775573, 0.5541376542, 0.4843495916, 0.4698741197, 0.1357579967, 0.4095643829,
		0.4481398626, 0.7113760662, 0.5559169994, 0.1942378494, 0.8298513232, 0.8320023342, 0.8056027821, 0.6379686778, 0.9194583965,
		0.9133343292, 0.8019644728, 0.5309330202, 0.0909256413, 0.1901014127, 0.3173854082, 0.8819429381, 0.1448828841, 0.3487725449,
		0.3777539067, 0.0917901759, 0.2960047437, 0.4549076388, 0.2813878898, 0.0929973445, 0.7324810918, 0.3799463829, 0.6087801987,
		0.7163824301, 0.2310489671, 0.6337259871, 0.3048684157, 0.3387296638, 0.8896184372, 0.8254166939, 0.4107967786, 0.1433201014,
		0.5807172221, 0.4941359194, 0.6047213719, 0.5814041459, 0.9407179140, 0.9939383765, 0.3564911426, 0.7716973153, 0.8175830588,
		0.2458311259, 0.6726336808, 0.8680681183, 0.2344609279, 0.2667416547, 0.7905403230, 0.1139956031, 0.7112792746, 0.5421166290,
		0.6555476101, 0.7982603464, 0.9427891524, 0.5630265226, 0.0529621550, 0.0767490955, 0.9764540804, 0.0229466953, 0.0357362313,
	};
	teq::CoordMap lhs([&indata](teq::MatrixT m)
		{
			for (teq::RankT i = 0; i < teq::mat_dim; ++i)
			{
				for (teq::RankT j = 0; j < teq::mat_dim; ++j)
				{
					m[i][j] = indata[i * teq::mat_dim + j];
				}
			}
		});
	teq::CoordMap rhs([&indata2](teq::MatrixT m)
		{
			for (teq::RankT i = 0; i < teq::mat_dim; ++i)
			{
				for (teq::RankT j = 0; j < teq::mat_dim; ++j)
				{
					m[i][j] = indata2[i * teq::mat_dim + j];
				}
			}
		});


	teq::MatrixT expected;
	for (teq::RankT i = 0; i < teq::mat_dim; ++i)
	{
		for (teq::RankT j = 0; j < teq::mat_dim; ++j)
		{
			expected[i][j] = 0;
			for (teq::RankT k = 0; k < teq::mat_dim; ++k)
			{
				expected[i][j] += indata[i * teq::mat_dim + k] * indata2[k * teq::mat_dim + j];
			}
		}
	}

	teq::iCoordMap* res = lhs.connect(rhs);
	res->access([&expected](const teq::MatrixT& m)
		{
			for (teq::RankT i = 0; i < teq::mat_dim; ++i)
			{
				for (teq::RankT j = 0; j < teq::mat_dim; ++j)
				{
					EXPECT_EQ(expected[i][j], m[i][j]);
				}
			}
		});

	delete res;
}


TEST(COORD, Reverse)
{
	std::vector<double> indata = {
		0.8156562350, 0.6115720004, 0.3259187100, 0.4587045151, 0.3708856605, 0.9116936271, 0.8506702867, 0.6005461299, 0.5781125392,
		0.2252200329, 0.6260416725, 0.7027031119, 0.8195474106, 0.9182331562, 0.4974968716, 0.3676863626, 0.4343532608, 0.8336493745,
		0.0609398686, 0.6964393063, 0.5466081949, 0.5578592341, 0.4815792839, 0.1296393464, 0.0192395135, 0.1068688839, 0.0634951350,
		0.5602241887, 0.7261901753, 0.8400658507, 0.8631530493, 0.1998541597, 0.6833077496, 0.2403264807, 0.6286744172, 0.8252406406,
		0.9807715918, 0.7744490835, 0.0027834805, 0.5948947457, 0.4003484346, 0.5155732371, 0.2212873196, 0.2115776178, 0.7312720364,
		0.7491175826, 0.8473974976, 0.5918676711, 0.4277359673, 0.0317686881, 0.7905199952, 0.4977264928, 0.0695145992, 0.5444856549,
		0.0884961428, 0.3255167258, 0.0738631045, 0.1542501963, 0.5192928180, 0.7640711766, 0.6548329341, 0.0041398851, 0.6030110971,
		0.2815274054, 0.4986186274, 0.4177728965, 0.5379782806, 0.9728425173, 0.8384357553, 0.1148654580, 0.6567525656, 0.4295099714,
		0.7243125736, 0.5892803635, 0.2597629537, 0.8375568398, 0.8496011347, 0.2944557711, 0.6930016846, 0.7247803680, 0.4856868442,
	};
	teq::CoordMap fwd([&indata](teq::MatrixT m)
		{
			for (teq::RankT i = 0; i < teq::mat_dim; ++i)
			{
				for (teq::RankT j = 0; j < teq::mat_dim; ++j)
				{
					m[i][j] = indata[i * teq::mat_dim + j];
				}
			}
		});

	teq::iCoordMap* rev = fwd.reverse();

	teq::MatrixT out;
	rev->access([&out, &fwd](const teq::MatrixT& bwd)
		{
			fwd.access([&out, &bwd](const teq::MatrixT& fwd)
			{
				teq::matmul(out, fwd, bwd);
			});
		});

	// expect matmul is identity
	for (teq::RankT i = 0; i < teq::mat_dim; ++i)
	{
		for (teq::RankT j = 0; j < teq::mat_dim; ++j)
		{
			if (i == j)
			{
				EXPECT_DOUBLE_EQ(1, std::round(out[i][j]));
			}
			else
			{
				EXPECT_DOUBLE_EQ(0, std::round(out[i][j]));
			}
		}
	}

	delete rev;
}


TEST(COORD, Identity)
{
	std::string idstr;
	teq::identity->access(
		[&](const teq::MatrixT& mat)
		{
			idstr = teq::to_string(mat);
		});

	std::string idstr2 = teq::identity->to_string();

	EXPECT_STREQ("[[1\\0\\0\\0\\0\\0\\0\\0\\0]\\\n"
		"[0\\1\\0\\0\\0\\0\\0\\0\\0]\\\n"
		"[0\\0\\1\\0\\0\\0\\0\\0\\0]\\\n"
		"[0\\0\\0\\1\\0\\0\\0\\0\\0]\\\n"
		"[0\\0\\0\\0\\1\\0\\0\\0\\0]\\\n"
		"[0\\0\\0\\0\\0\\1\\0\\0\\0]\\\n"
		"[0\\0\\0\\0\\0\\0\\1\\0\\0]\\\n"
		"[0\\0\\0\\0\\0\\0\\0\\1\\0]\\\n"
		"[0\\0\\0\\0\\0\\0\\0\\0\\1]]", idstr.c_str());
	EXPECT_STREQ(idstr.c_str(), idstr2.c_str());

	teq::CoordT fwd_out;
	teq::CoordT icoord = {
		42, 12, 85, 7, 82, 91, 2, 34,
	};

	teq::identity->forward(fwd_out.begin(), icoord.begin());
	EXPECT_ARREQ(icoord, fwd_out);
}


TEST(COORD, Reduce)
{
	size_t rank = 5;
	std::vector<teq::DimT> red = {22, 32, 2};
	teq::CoordptrT reducer = teq::reduce(rank, red);

	teq::CoordT fwd_out;
	teq::CoordT icoord = {
		211.6172349153, 3.6941314330, 3.3471187148, 24.3511302088,
		17.8520169468, 99.9911659058, 7.2182000783, 6.4776819746
	};

	reducer->forward(fwd_out.begin(), icoord.begin());
	for (size_t i = 0; i < rank; ++i)
	{
		EXPECT_EQ(icoord[i], fwd_out[i]) << i;
	}
	for (size_t i = rank; i < teq::rank_cap; ++i)
	{
		EXPECT_DOUBLE_EQ(icoord[i] / red[i - rank], fwd_out[i]) << "red=" << red[i - rank] << ",i=" << i;
	}

	EXPECT_FATAL(teq::reduce(rank, {0}), "cannot reduce using zero dimensions [0]");

	std::string fatalmsg = fmts::sprintf(
		"cannot reduce shape rank %d beyond rank_cap with n_red %d",
		rank + 1, red.size());
	EXPECT_FATAL(teq::reduce(rank + 1, red), fatalmsg.c_str());

	EXPECT_WARN(teq::reduce(0, {}), "reducing scalar ... will do nothing");
}


TEST(COORD, Extend)
{
	size_t rank = 3;
	std::vector<teq::DimT> ext = {12, 21, 8, 4, 52};
	teq::CoordptrT extender = teq::extend(rank, ext);

	teq::CoordT fwd_out;
	teq::CoordT incoord = {142.2, 42.17, 33.292, 33, 231.5, 2.33, 96.4, 1.23};

	extender->forward(fwd_out.begin(), incoord.begin());
	for (size_t i = 0; i < rank; ++i)
	{
		EXPECT_EQ(incoord[i], fwd_out[i]) << i;
	}
	for (size_t i = rank; i < teq::rank_cap; ++i)
	{
		EXPECT_DOUBLE_EQ(incoord[i] * ext[i - rank], fwd_out[i]) << "ext=" << ext[i - rank] << ",i=" << i;
	}

	EXPECT_FATAL(teq::extend(rank, {0}), "cannot extend using zero dimensions [0]");

	std::string fatalmsg = fmts::sprintf(
		"cannot extend shape rank %d beyond rank_cap with n_ext %d",
		rank + 1, ext.size());
	EXPECT_FATAL(teq::extend(rank + 1, ext), fatalmsg.c_str());

	EXPECT_WARN(teq::extend(0, {}), "extending with empty vector ... will do nothing");
}


TEST(COORD, Permute)
{
	std::vector<teq::RankT> perm = {4, 2, 3, 7, 0, 1};
	teq::CoordptrT permuter = teq::permute(perm);
	std::array<bool,teq::rank_cap> permed;
	permed.fill(false);
	for (teq::DimT p : perm)
	{
		permed[p] = true;
	}
	for (size_t i = 0; i < teq::rank_cap; ++i)
	{
		if (false == permed[i])
		{
			perm.push_back(i);
		}
	}

	teq::CoordT fwd_out;
	teq::CoordT icoord = {12, 82, 20, 31, 49, 1, 1, 1};

	permuter->forward(fwd_out.begin(), icoord.begin());
	for (size_t i = 0; i < teq::rank_cap; ++i)
	{
		EXPECT_EQ(icoord[perm[i]], fwd_out[i]);
	}

	EXPECT_WARN(teq::permute({}), "permuting with same dimensions ... will do nothing");
}


TEST(COORD, Flip)
{
	size_t dim = 3;
	teq::CoordptrT flipper = teq::flip(dim);

	teq::CoordT fwd_out;
	teq::CoordT icoord = {23, 66, 72, 83, 91, 1, 31, 21};

	flipper->forward(fwd_out.begin(), icoord.begin());
	for (size_t i = 0; i < dim; ++i)
	{
		EXPECT_EQ(icoord[i], fwd_out[i]) << i;
	}
	for (size_t i = dim + 1; i < teq::rank_cap; ++i)
	{
		EXPECT_EQ(icoord[i], fwd_out[i]) << i;
	}
	EXPECT_EQ(-icoord[dim]-1, fwd_out[dim]);

	EXPECT_WARN(teq::flip(teq::rank_cap * 2), "flipping dimension out of rank_cap ... will do nothing");
}


TEST(COORD, Bijection)
{
	EXPECT_TRUE(teq::identity->is_bijective());

	teq::CoordptrT reducer = teq::reduce(3, {2});
	EXPECT_FALSE(reducer->is_bijective());
}


TEST(COORD, IsIdentity)
{
	EXPECT_TRUE(teq::is_identity(nullptr));
	EXPECT_TRUE(teq::is_identity(teq::identity.get()));

	teq::CoordptrT sample_id = std::make_shared<teq::CoordMap>(
		*static_cast<teq::CoordMap*>(teq::identity.get())); // deep copy
	EXPECT_TRUE(teq::is_identity(sample_id.get()));

	teq::CoordptrT bourne(new teq::CoordMap(
		[](teq::MatrixT fwd)
		{
			// todo: we can randomize this so long as fwd is not identity
			for (teq::RankT i = 0; i < teq::rank_cap; ++i)
			{
				fwd[i][i] = 2;
			}
		}));
	EXPECT_FALSE(teq::is_identity(bourne.get()));
}


#endif // DISABLE_COORD_TEST