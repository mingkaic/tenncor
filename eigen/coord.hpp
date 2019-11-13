///
/// coord.hpp
/// eigen
///
/// Purpose:
/// Define Eigen transformation argument wrapper
///

#include <functional>
#include <set>

#include "eigen/matops.hpp"

#ifndef EIGEN_COORD_HPP
#define EIGEN_COORD_HPP

namespace eigen
{

struct CoordMap final
{
	CoordMap (teq::CoordT indices)
	{
		std::fill(args_[0], args_[0] + teq::mat_size, std::nan(""));
		for (size_t i = 0; i < teq::rank_cap; ++i)
		{
			args_[0][i] = indices[i];
		}
	}

	// todo: make init safer (values between rows are nan)
	CoordMap (teq::MatInitF init)
	{
		std::fill(args_[0], args_[0] + teq::mat_size, std::nan(""));
		init(args_);
	}

	/// Return string representation of shape mapper
	std::string to_string (void) const
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

	/// Access the forward matrix representation of transformer as a param to input
	/// callback function cb
	void access (std::function<void(const teq::MatrixT&)> cb) const
	{
		cb(args_);
	}

private:
	teq::MatrixT args_;
};

/// Type of coordinate mapping smartpointer
using CoordptrT = std::shared_ptr<CoordMap>;

/// Return CoordMap wrapper of reduction dimensions
CoordptrT reduce (std::set<teq::RankT> red_dims);

/// Return CoordMap wrapper of extension parameters
CoordptrT extend (teq::CoordT bcast);

/// Return CoordMap wrapper of permute indices
CoordptrT permute (std::array<teq::RankT,teq::rank_cap> dims);

}

#endif // EIGEN_COORD_HPP
