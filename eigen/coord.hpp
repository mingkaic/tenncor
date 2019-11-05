///
/// coord.hpp
/// eigen
///
/// Purpose:
/// Define Eigen transformation argument wrapper
///

#include "teq/coord.hpp"

#ifndef EIGEN_COORD_HPP
#define EIGEN_COORD_HPP

namespace eigen
{

/// Eigen transformation wrapper implementation of iCoordMap
// todo: replace this with teq::CoordMap
struct CoordMap final : public teq::iCoordMap
{
	CoordMap (teq::CoordT indices, bool bijective = false) :
		bijective_(bijective)
	{
		std::fill(args_[0], args_[0] + teq::mat_size, std::nan(""));
		for (size_t i = 0; i < teq::rank_cap; ++i)
		{
			args_[0][i] = indices[i];
		}
	}

	// todo: make init safer (values between rows are nan)
	CoordMap (teq::MatInitF init, bool bijective = false) :
		bijective_(bijective)
	{
		std::fill(args_[0], args_[0] + teq::mat_size, std::nan(""));
		init(args_);
	}

	/// Implementation of iCoordMap
	teq::iCoordMap* connect (const teq::iCoordMap& rhs) const override
	{
		return nullptr;
	}

	/// Implementation of iCoordMap
	void forward (teq::CoordT::iterator out,
		teq::CoordT::const_iterator in) const override {}

	/// Implementation of iCoordMap
	iCoordMap* reverse (void) const override
	{
		return nullptr;
	}

	/// Implementation of iCoordMap
	std::string to_string (void) const override
	{
		std::stringstream ss;
		ss << fmts::arr_begin;
		if (false == std::isnan(args_[0][0]))
		{
			ss << fmts::arr_begin << args_[0][0];
			for (teq::RankT j = 1; j < teq::mat_dim &&
				false == std::isnan(args_[0][j]); ++j)
			{
				ss << fmts::arr_delim << args_[0][j];
			}
			ss << fmts::arr_end;
			for (teq::RankT i = 1; i < teq::mat_dim &&
				false == std::isnan(args_[i][0]); ++i)
			{
				ss << fmts::arr_delim << '\n' << fmts::arr_begin << args_[i][0];
				for (teq::RankT j = 1; j < teq::mat_dim &&
					false == std::isnan(args_[i][j]); ++j)
				{
					ss << fmts::arr_delim << args_[i][j];
				}
				ss << fmts::arr_end;
			}
		}
		ss << fmts::arr_end;
		return ss.str();
	}

	/// Implementation of iCoordMap
	void access (std::function<void(const teq::MatrixT&)> cb) const override
	{
		cb(args_);
	}

	/// Implementation of iCoordMap
	bool is_bijective (void) const override
	{
		return bijective_;
	}

private:
	teq::MatrixT args_;

	bool bijective_;
};

/// Type of iCoordMap smartpointer
using CoordptrT = std::shared_ptr<CoordMap>;

/// Return CoordMap wrapper of reduction dimensions
CoordptrT reduce (std::vector<teq::RankT> red_dims);

/// Return CoordMap wrapper of extension parameters
CoordptrT extend (teq::RankT rank, std::vector<teq::DimT> ext);

/// Return CoordMap wrapper of permute indices
CoordptrT permute (std::vector<teq::RankT> dims);

}

#endif // EIGEN_COORD_HPP
