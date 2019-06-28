
#ifndef DISABLE_MATOPS_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "ade/matops.hpp"


TEST(MATOPS, ToString)
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


TEST(MATOPS, Determinant)
{
	ade::MatrixT indata = {
		{0.6889268247, 0.5182375525, 0.8077819453, 0.6586822856, 0.1064583106, 0.5584794867, 0.7151236734, 0.6955541292, 0.5299556786},
		{0.8790203846, 0.7084063896, 0.0299713280, 0.6855684925, 0.1025625016, 0.8161998141, 0.1896595999, 0.1917321581, 0.3208859474},
		{0.5316041393, 0.2931242636, 0.1538743813, 0.8752116402, 0.3936777315, 0.2557857831, 0.0779308587, 0.2442668717, 0.0513855996},
		{0.8207417781, 0.6037351410, 0.5996222865, 0.6397019876, 0.6966817452, 0.1057642881, 0.5204656744, 0.4548947579, 0.2431977077},
		{0.0085720013, 0.7828366981, 0.8081878276, 0.9939625759, 0.0395092265, 0.1779490257, 0.5277815389, 0.3918598499, 0.1464174550},
		{0.4013652456, 0.4390466271, 0.0364222084, 0.2707301631, 0.2836227355, 0.7225988639, 0.3751309759, 0.2525261161, 0.6025458644},
		{0.2754987472, 0.0453902142, 0.1057272602, 0.0953509452, 0.4041199156, 0.0549068665, 0.5641598371, 0.4743610631, 0.1015215133},
		{0.2252390787, 0.3721492337, 0.2498504751, 0.3123788027, 0.5537316928, 0.5611137585, 0.2029485252, 0.8579680347, 0.0442760832},
		{0.3058592439, 0.9815336218, 0.3364270850, 0.1562163045, 0.5562589952, 0.3769814342, 0.4465301119, 0.6977257625, 0.7664397080},
	};

	EXPECT_DOUBLE_EQ(-0.028472153084121096, ade::determinant(indata));

	ade::MatrixT determ0 = {
		{0.6889268247, 0, 0.8077819453, 0.6586822856, 0.1064583106, 0.5584794867, 0.7151236734, 0.6955541292, 0.5299556786},
		{0.8790203846, 0, 0.0299713280, 0.6855684925, 0.1025625016, 0.8161998141, 0.1896595999, 0.1917321581, 0.3208859474},
		{0.5316041393, 0.2931242636, 0.1538743813, 0.8752116402, 0, 0.2557857831, 0.0779308587, 0.2442668717, 0.0513855996},
		{0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0.0085720013, 0.7828366981, 0.8081878276, 0.9939625759, 0.0395092265, 0.1779490257, 0.5277815389, 0.3918598499, 0.1464174550},
		{0.4013652456, 0.4390466271, 0.0364222084, 0.2707301631, 0.2836227355, 0.7225988639, 0.3751309759, 0.2525261161, 0.6025458644},
		{0.2754987472, 0.0453902142, 0.1057272602, 0.0953509452, 0.4041199156, 0.0549068665, 0.5641598371, 0.4743610631, 0.1015215133},
		{0.2252390787, 0.3721492337, 0.2498504751, 0.3123788027, 0.5537316928, 0.5611137585, 0.2029485252, 0.8579680347, 0.0442760832},
		{0.3058592439, 0.9815336218, 0.3364270850, 0.1562163045, 0.5562589952, 0.3769814342, 0.4465301119, 0.6977257625, 0.7664397080},
	};

	EXPECT_DOUBLE_EQ(0, ade::determinant(determ0));
}


TEST(MATOPS, Inverse)
{
	ade::MatrixT out, in;
	ade::MatrixT zout, zin;
	ade::MatrixT badout, badin;
	std::vector<double> indata = {
		0.6889268247, 0.5182375525, 0.8077819453, 0.6586822856, 0.1064583106, 0.5584794867, 0.7151236734, 0.6955541292, 0.5299556786,
		0.8790203846, 0.7084063896, 0.0299713280, 0.6855684925, 0.1025625016, 0.8161998141, 0.1896595999, 0.1917321581, 0.3208859474,
		0.5316041393, 0.2931242636, 0.1538743813, 0.8752116402, 0.3936777315, 0.2557857831, 0.0779308587, 0.2442668717, 0.0513855996,
		0.8207417781, 0.6037351410, 0.5996222865, 0.6397019876, 0.6966817452, 0.1057642881, 0.5204656744, 0.4548947579, 0.2431977077,
		0.0085720013, 0.7828366981, 0.8081878276, 0.9939625759, 0.0395092265, 0.1779490257, 0.5277815389, 0.3918598499, 0.1464174550,
		0.4013652456, 0.4390466271, 0.0364222084, 0.2707301631, 0.2836227355, 0.7225988639, 0.3751309759, 0.2525261161, 0.6025458644,
		0.2754987472, 0.0453902142, 0.1057272602, 0.0953509452, 0.4041199156, 0.0549068665, 0.5641598371, 0.4743610631, 0.1015215133,
		0.2252390787, 0.3721492337, 0.2498504751, 0.3123788027, 0.5537316928, 0.5611137585, 0.2029485252, 0.8579680347, 0.0442760832,
		0.3058592439, 0.9815336218, 0.3364270850, 0.1562163045, 0.5562589952, 0.3769814342, 0.4465301119, 0.6977257625, 0.7664397080
	};
	std::vector<double> zdata = {
		0.2878212480, 0.9675515366, 0.4965917427, 0.3427207542, 0.3951758902, 0.9356175241, 0.5522683328, 0.8351937923, 0.3580548585,
		0.2693744059, 0.9866321188, 0.7443338139, 0.3932269328, 0.3664260724, 0.5003604686, 0.8841808721, 0.6666851450, 0.8635718932,
		0, 0.5992231635, 0.2092168782, 0.9006493407, 0.1660514203, 0.3211276588, 0.0488621576, 0.4741749870, 0.6866121020,
		0.1397692183, 0.4307816850, 0.8882817006, 0.1124489938, 0.9227742019, 0.8230034221, 0.3495727035, 0.1150046140, 0.1994296875,
		0, 0, 0, 0.2337627905, 0.9459128753, 0.7781876174, 0.7045714368, 0.7501464714, 0.3326616545,
		0, 0, 0.4754745317, 0.6012895807, 0.9762103119, 0.0438381746, 0.5144917415, 0.7792010320, 0.4599751149,
		0.3482648273, 0.4295310003, 0.5859607927, 0.0108565503, 0.8980491091, 0.9676011388, 0.0798446124, 0.2528632391, 0.4186041972,
		0, 0.0186201451, 0.2775045823, 0.8535831918, 0.0246002084, 0.8463399072, 0.7754453131, 0.9828614649, 0.1672314745,
		0.1414881481, 0.0058712489, 0.9803878600, 0.8148616354, 0.4505866891, 0.3767108235, 0.2663544360, 0.0881545477, 0.3644870769
	};
	std::vector<double> baddata = {
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		0.8790203846, 0.7084063896, 0.0299713280, 0.6855684925, 0.1025625016, 0.8161998141, 0.1896595999, 0.1917321581, 0.3208859474,
		0.5316041393, 0.2931242636, 0.1538743813, 0.8752116402, 0.3936777315, 0.2557857831, 0.0779308587, 0.2442668717, 0.0513855996,
		0.8207417781, 0.6037351410, 0.5996222865, 0.6397019876, 0.6966817452, 0.1057642881, 0.5204656744, 0.4548947579, 0.2431977077,
		0.0085720013, 0.7828366981, 0.8081878276, 0.9939625759, 0.0395092265, 0.1779490257, 0.5277815389, 0.3918598499, 0.1464174550,
		0.4013652456, 0.4390466271, 0.0364222084, 0.2707301631, 0.2836227355, 0.7225988639, 0.3751309759, 0.2525261161, 0.6025458644,
		0.2754987472, 0.0453902142, 0.1057272602, 0.0953509452, 0.4041199156, 0.0549068665, 0.5641598371, 0.4743610631, 0.1015215133,
		0.2252390787, 0.3721492337, 0.2498504751, 0.3123788027, 0.5537316928, 0.5611137585, 0.2029485252, 0.8579680347, 0.0442760832,
		0.3058592439, 0.9815336218, 0.3364270850, 0.1562163045, 0.5562589952, 0.3769814342, 0.4465301119, 0.6977257625, 0.7664397080
	};
	for (uint8_t i = 0; i < ade::mat_dim; ++i)
	{
		for (uint8_t j = 0; j < ade::mat_dim; ++j)
		{
			in[i][j] = indata[i * ade::mat_dim + j];
			zin[i][j] = zdata[i * ade::mat_dim + j];
			badin[i][j] = baddata[i * ade::mat_dim + j];
		}
	}

	ade::inverse(out, in);
	ade::inverse(zout, zin);

	std::string fatalmsg = fmts::sprintf("cannot invert matrix:\n%s",
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


TEST(MATOPS, Matmul)
{
	ade::MatrixT expected, out, in, in2;
	std::vector<double> indata = {
		0.7259523165, 0.0215138058, 0.8619883459, 0.2181517503, 0.4143879487, 0.3798615637, 0.9794909452, 0.8138574826, 0.6938136856,
		0.1179236527, 0.8269392333, 0.7848566651, 0.4425554320, 0.2254599610, 0.2829998577, 0.4309979629, 0.4136898807, 0.0993291362,
		0.3851670356, 0.6924332537, 0.7581359080, 0.9919464641, 0.9373232736, 0.9972001298, 0.5470192919, 0.8030033044, 0.6586853820,
		0.4327419661, 0.6736792920, 0.7152885302, 0.2631893982, 0.0667248482, 0.9053717683, 0.4811290984, 0.3568605660, 0.1108425822,
		0.8035430215, 0.2026908424, 0.7012492508, 0.9705152395, 0.4908655548, 0.9507130677, 0.1450518113, 0.9853334787, 0.5380276943,
		0.6780259470, 0.1551816358, 0.2960734837, 0.9236366285, 0.8219052133, 0.0704425563, 0.1578115478, 0.5913271639, 0.7263275783,
		0.9744008506, 0.7068900791, 0.3345331636, 0.5890527786, 0.5976463550, 0.0837041937, 0.4348178698, 0.4044658578, 0.8234606863,
		0.7836847648, 0.2950285389, 0.2692438132, 0.6663220699, 0.9641675815, 0.4014375657, 0.5141888657, 0.2204340247, 0.4188736933,
		0.5540695317, 0.2548572721, 0.6360068068, 0.0446748122, 0.1895554973, 0.5437636026, 0.1918515441, 0.1516492371, 0.2869706044
	};
	std::vector<double> indata2 = {
		0.3300882219, 0.8257034380, 0.7968153097, 0.9297766417, 0.7073528964, 0.0358232206, 0.5658120433, 0.0957130274, 0.8546985226,
		0.5444767126, 0.0581139125, 0.0470172441, 0.9976957098, 0.0561366667, 0.3895200681, 0.5278496786, 0.3867032128, 0.8442186432,
		0.0100940012, 0.5477036787, 0.6507459728, 0.1544754538, 0.7275259686, 0.0982912445, 0.8102959851, 0.2520050806, 0.8211539629,
		0.7415426779, 0.2065689307, 0.1473578853, 0.3859136732, 0.7126693468, 0.1215760994, 0.0403584690, 0.8706772052, 0.9182623904,
		0.7294302408, 0.7466965934, 0.6277598021, 0.1270593047, 0.7527522045, 0.1423606412, 0.6317469358, 0.7519172313, 0.1923481392,
		0.1047496241, 0.9343753287, 0.9727614615, 0.8273784528, 0.9352531933, 0.8941779514, 0.3018683143, 0.2000405141, 0.5363161670,
		0.3884492851, 0.7588151057, 0.9933565098, 0.0970409864, 0.1252498267, 0.3062819161, 0.1766142120, 0.3591992015, 0.0084369594,
		0.2479436504, 0.1266563150, 0.5884766852, 0.5278691634, 0.8695170450, 0.6803630612, 0.0420603430, 0.3344162464, 0.8860979790,
		0.8208910642, 0.2098140918, 0.3228215729, 0.7025393155, 0.8154172074, 0.4992314192, 0.3955948624, 0.0212939634, 0.1960803287
	};
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
