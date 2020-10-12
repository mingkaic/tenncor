
#ifndef GLOBAL_MOCK_GENERATOR_HPP
#define GLOBAL_MOCK_GENERATOR_HPP

#include "internal/global/global.hpp"

struct MockGenerator : public global::iGenerator
{
	std::string get_str (void) const override
	{
		return fmts::to_string(++counter_);
	}

	int64_t unif_int (
		const int64_t& lower, const int64_t& upper) const override
	{
		return randomizer_.unif_int(lower, upper);
	}

	double unif_dec (
		const double& lower, const double& upper) const override
	{
		return randomizer_.unif_dec(lower, upper);
	}

	double norm_dec (
		const double& mean, const double& stdev) const override
	{
		return randomizer_.norm_dec(mean, stdev);
	}

	global::GenF<std::string> get_strgen (void) const override
	{
		return [this](){ return get_str(); };
	}

	global::GenF<int64_t> unif_intgen (
		const int64_t& lower, const int64_t& upper) const override
	{
		return randomizer_.unif_intgen(lower, upper);
	}

	global::GenF<double> unif_decgen (
		const double& lower, const double& upper) const override
	{
		return randomizer_.unif_decgen(lower, upper);
	}

	global::GenF<double> norm_decgen (
		const double& mean, const double& stdev) const override
	{
		return randomizer_.norm_decgen(mean, stdev);
	}

private:
	mutable size_t counter_ = 0;

	global::Randomizer randomizer_;
};

#endif // GLOBAL_MOCK_GENERATOR_HPP
