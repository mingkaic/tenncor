//
// Created by Mingkai Chen on 2017-04-18.
//

#include "tests/include/utils/fuzz.h"

#ifdef TENNCOR_FUZZ_H

namespace FUZZ
{

void fuzz_test::SetUp (void) {}

void fuzz_test::TearDown (void)
{
	if (::testing::Test::HasFailure())
	{
		std::cout << "FUZZ OUTPUT:" << std::endl << ss_.str() << std::endl;
	}
	ss_.str("");
	ss_.clear();
}

std::vector<double> fuzz_test::get_double (size_t len,
	std::string purpose,
	std::pair<double,double> range)
{
	double min, max;
	if (range.first == range.second)
	{
		min = std::numeric_limits<double>::min();
		max = std::numeric_limits<double>::max();
	}
	else
	{
		min = range.first;
		max = range.second;
	}

	std::vector<double> vec;
	std::uniform_real_distribution<double> dis(min, max);
	std::default_random_engine& generator = nnutils::get_generator();

	ss_ << purpose << ": double<";
	for (size_t i = 0; i < len; i++)
	{
		double val = dis(generator);
		ss_ << val<< ",";
		vec.push_back(val);
	}
	ss_ << ">" << std::endl;

	return vec;
}

std::vector<size_t> fuzz_test::get_int (size_t len,
	std::string purpose,
	std::pair<size_t,size_t> range)
{
	size_t min, max;
	if (range.first == range.second)
	{
		min = std::numeric_limits<size_t>::min();
		max = std::numeric_limits<size_t>::max();
	}
	else
	{
		min = range.first;
		max = range.second;
	}

	std::vector<size_t> vec;
	std::uniform_int_distribution<size_t> dis(min, max);
	std::default_random_engine& generator = nnutils::get_generator();
	
	ss_ << purpose << ": int<";
	for (size_t i = 0; i < len; i++)
	{
		size_t val = dis(generator);
		ss_ << val << ",";
		vec.push_back(val);
	}
	ss_ << ">" << std::endl;

	return vec;
}

std::string fuzz_test::get_string (size_t len,
	std::string purpose,
	std::string alphanum)
{
	std::vector<size_t> indices = get_int(len, "indices", {0, alphanum.size()-1});
	std::string s(len, ' ');
	std::transform(indices.begin(), indices.end(), s.begin(),
	[&alphanum](size_t index)
	{
		return alphanum[index];
	});
	ss_ << purpose << ": string<" << s << ">" << std::endl;

	return s;
}

}

#endif