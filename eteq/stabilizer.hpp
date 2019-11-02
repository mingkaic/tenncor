#include "eteq/generated/opcode.hpp"
#include "eteq/inode.hpp"

#ifndef ETEQ_STABILIZER_HPP
#define ETEQ_STABILIZER_HPP

namespace eteq
{

template <typename T>
using NumRanges = std::vector<estd::NumRange<T>>;

template <typename T>
T sigmoid (const T& arg)
{
	if (arg < -4)
	{
		return 0;
	}
	else if (arg > 4)
	{
		return 1;
	}
	return 1 / (1 + std::exp(-arg));
}

template <typename T>
T tanh (const T& arg)
{
	if (arg < -4)
	{
		return -1;
	}
	else if (arg > 4)
	{
		return 1;
	}
	return std::tanh(arg);
}

template <typename T>
T cube (const T& arg)
{
	return arg * arg * arg;
}

template <typename T>
estd::NumRange<T> generate_range (teq::iFunctor* func, const NumRanges& ranges)
{
	teq::Opcode opcode = func->get_opcode();
	switch (opcode.code_)
	{
		case egen::ABS:
		{
			T lower = ranges[0].lower_;
			T upper = ranges[0].upper_;
			if (lower < 0 && upper > 0)
			{
				// original range crosses 0
				lower = 0;
			}
			else
			{
				lower = std::abs(lower);
				upper = std::abs(upper);
			}
			return estd::NumRange<T>(lower, upper);
		}
		case egen::NEG:
			return estd::NumRange<T>(-ranges[0].lower_, -ranges[0].upper_);
		case egen::SIN:
		{
			double pi = boost::math::constants::pi<double>();
			T lower = ranges[0].lower_;
			T upper = ranges[0].upper_;

			T diff = upper - lower;
			if (diff < pi * 2)
			{
				T ql = std::ceil(lower - (pi / 2) / pi) * pi;
				T qu = std::floor(upper - (pi / 2) / pi) * pi;
				T sl = std::sin(lower);
				T su = std::sin(upper);
				if (qu < ql) // there is no extrema between ranges
				{
					return estd::NumRange<T>(sl, su);
				}
				// cos(ql) is the nearest range extrema above lower
				// cos(qu) is the nearest range extrema below upper
				std::vector<T> bounds = {std::cos(ql), std::cos(qu), sl, su};
				return estd::NumRange<T>(
					*std::min_element(bounds.begin(), bounds.end()),
					*std::max_element(bounds.begin(), bounds.end()))
			}
			return estd::NumRange<T>(-1, 1);
		}
		case egen::COS:
		{
			double pi = boost::math::constants::pi<double>();
			T lower = ranges[0].lower_;
			T upper = ranges[0].upper_;

			T diff = upper - lower;
			if (diff < pi * 2)
			{
				T ql = std::ceil(lower / pi) * pi;
				T qu = std::floor(upper / pi) * pi;
				T sl = std::cos(lower);
				T su = std::cos(upper);
				if (qu < ql) // there is no extrema between ranges
				{
					return estd::NumRange<T>(sl, su);
				}
				// cos(ql) is the nearest range extrema above lower
				// cos(qu) is the nearest range extrema below upper
				std::vector<T> bounds = {std::cos(ql), std::cos(qu), sl, su};
				return estd::NumRange<T>(
					*std::min_element(bounds.begin(), bounds.end()),
					*std::max_element(bounds.begin(), bounds.end()));
			}
			return estd::NumRange<T>(-1, 1);
		}
		case egen::TAN:
		{
			double pi = boost::math::constants::pi<double>();
			T lower = ranges[0].lower_;
			T upper = ranges[0].upper_;

			if (std::round(lower / pi) * pi == std::round(upper / pi) * pi)
			{
				// ranges are within finite values
				return estd::NumRange<T>(std::tanh(lower), std::tanh(upper));
			}
			T inf = std::numeric_limits<T>::infinity();
			return estd::NumRange<T>(-inf, inf);
		}
		case egen::EXP:
			return estd::NumRange<T>(std::exp(ranges[0].lower_), std::exp(ranges[0].upper_));
		case egen::LOG:
			return estd::NumRange<T>(std::log(ranges[0].lower_), std::log(ranges[0].upper_));
		case egen::SQRT:
			return estd::NumRange<T>(std::sqrt(ranges[0].lower_), std::sqrt(ranges[0].upper_));
		case egen::ROUND:
			return estd::NumRange<T>(std::round(ranges[0].lower_), std::round(ranges[0].upper_));
		case egen::SQUARE:
		{
			T lower = ranges[0].lower_;
			T upper = ranges[0].upper_;
			return estd::NumRange<T>(
				lower < 0 && upper > 0 ? 0 : lower * lower,
				upper * upper);
		}
		case egen::CUBE:
			return estd::NumRange<T>(cube(ranges[0].lower_), cube(ranges[0].upper_));
		case egen::SIGMOID:
			return estd::NumRange<T>(sigmoid(ranges[0].lower_), sigmoid(ranges[0].upper_));
		case egen::TANH:
			return estd::NumRange<T>(tanh(ranges[0].lower_), tanh(ranges[0].upper_));
		case egen::PERMUTE:
		case egen::EXTEND:
		case egen::RESHAPE:
		case egen::REVERSE:
		case egen::RAND_UNIF:
		case egen::SLICE:
		case egen::STRIDE:
			return ranges[0];
		case egen::PAD:
		case egen::SCATTER:
		{
			std::vector<T> bounds = {ranges[0].lower_, ranges[0].upper_, 0};
			return estd::NumRange<T>(
				*std::min_element(bounds.begin(), bounds.end()),
				*std::max_element(bounds.begin(), bounds.end()));
		}
		case egen::ARGMAX:
		{
			auto& arg = func->get_children()[0];
			auto coorder = arg.get_coorder();
			assert(coorder != nullptr);
			teq::RankT return_dim;
			coorder->access(
				[&](const teq::MatrixT& args)
				{
					return_dim = args[0][0];
				});
			teq::Shape shape = arg.get_tensor()->shape();
			return estd::NumRange<T>(0, teq::rank_cap == return_dim ?
				shape.n_elems() : shape.at(return_dim));
		}
		case egen::SELECT:
		{
			std::vector<T> lbounds, ubounds;
			lbounds.reserve(ranges.size() - 1);
			ubounds.reserve(ranges.size() - 1);
			for (size_t i = 1, n = ranges.size(); i < n; ++i)
			{
				lbounds.push_back(ranges[i].lower_);
				ubounds.push_back(ranges[i].upper_);
			}
			return estd::NumRange<T>(
				*std::min_element(lbounds.begin(), lbounds.end()),
				*std::max_element(ubounds.begin(), ubounds.end()));
		}
		case egen::CONCAT:
		case egen::GROUP_CONCAT:
		{
			std::vector<T> lbounds, ubounds;
			lbounds.reserve(ranges.size());
			ubounds.reserve(ranges.size());
			for (auto& range : ranges)
			{
				lbounds.push_back(range.lower_);
				ubounds.push_back(range.upper_);
			}
			return estd::NumRange<T>(
				*std::min_element(lbounds.begin(), lbounds.end()),
				*std::max_element(ubounds.begin(), ubounds.end()));
		}
		case egen::POW:
		{
			T lower_base = ranges[0].lower_;
			T upper_base = ranges[0].upper_;
			T lower_exp = ranges[1].lower_;
			T upper_exp = ranges[1].upper_;
			if (lower_exp == upper_exp || lower_base == upper_base)
			{
				return estd::NumRange<T>(
					std::pow(lower_base, lower_exp),
					std::pow(upper_base, upper_exp));
			}
			T upper = std::pow(upper_base, upper_exp);
			T lower = lower_base;
			if (lower < 0)
			{
				if (std::is_integral<T>::value)
				{
					// exponent can't be decimal
					if (0 == lower_exp % 2)
					{
						// look for nearest odd for lowest
						lower = std::pow(lower, lower_exp + 1);
						upper = std::max(upper, std::pow(lower, lower_exp));
					}
					else
					{
						lower = std::pow(lower, lower_exp);
						// look for nearest even for highest
						upper = std::max(upper, std::pow(lower, lower_exp + 1));
					}
				}
				else
				{
					// exponent be a decimal and base can be negative
					lower = std::nan();
					lower_exp = std::ceil(lower_exp);
					// look for nearest even for highest
					upper = std::max(upper, std::pow(lower,
						lower_exp % 2 ? lower_exp + 1 : lower_exp));
				}
			}
			else
			{
				// base is purely positive
				std::vector<T> bounds = {upper,
					std::pow(lower_base, lower_exp),
					std::pow(lower_base, upper_exp),
					std::pow(upper_base, lower_exp)};
				lower = *std::min_element(bounds.begin(), bounds.end());
				upper = *std::max_element(bounds.begin(), bounds.end());
			}
			return estd::NumRange<T>(lower, upper);
		}
		case egen::ADD:
			return estd::NumRange<T>(
				ranges[0].lower_ + ranges[1].lower_,
				ranges[0].upper_ + ranges[1].upper_);
		case egen::SUB:
			return estd::NumRange<T>(
				ranges[0].lower_ + ranges[1].upper_,
				ranges[0].upper_ + ranges[1].lower_);
		case egen::MUL:
		{
			T alower = ranges[0].lower_;
			T aupper = ranges[0].upper_;
			T blower = ranges[1].lower_;
			T bupper = ranges[1].upper_;
			std::vector<T> bounds = {
				alower * blower,
				alower * bupper,
				aupper * blower,
				aupper * bupper};
			return estd::NumRange<T>(
				*std::min_element(bounds.begin(), bounds.end()),
				*std::max_element(bounds.begin(), bounds.end()));
		}
		case egen::DIV:
		case egen::MIN:
		case egen::MAX:
		case egen::EQ:
		case egen::NEQ:
		case egen::LT:
		case egen::GT:
		case egen::REDUCE_SUM:
		case egen::REDUCE_PROD:
		case egen::REDUCE_MIN:
		case egen::REDUCE_MAX:
		case egen::MATMUL:
		case egen::CONV:
			logs::fatal("not yet implemented");
		default:
			break;
	}
	return estd::NumRange<T>();
}

// todo: handle complex T
template <typename T>
struct Stabilizer final : public teq::iTraveler
{
	/// Implementation of iTraveler
	void visit (teq::iLeaf* leaf) override
	{
		if (false == estd::has(ranges_, leaf))
		{
			auto dleaf = dynamic_cast<eteq::iLeaf<T>*>(leaf);
			if (nullptr != dleaf && dleaf->is_const())
			{
				auto data = (T*) dleaf->data();
				teq::NElemT n = dleaf->shape().n_elems();
				ranges_.emplace(leaf, estd::NumRange<T>(
					*std::min_element(data, data + n),
					*std::max_element(data, data + n)));
			}
			else
			{
				ranges_.emplace(leaf, estd::NumRange<T>(
					std::numeric_limits<T>::min(),
					std::numeric_limits<T>::max()));
			}
		}
	}

	/// Implementation of iTraveler
	void visit (teq::iFunctor* func) override
	{
		if (false == estd::has(ranges_, func))
		{
			auto& args = func->get_children();
			NumRanges<T> ranges;
			ranges.reserve(args.size());
			for (auto& arg : args)
			{
				teq::iTensor* argtens = arg.get_tensor().get();
				argtens->accept(*this);
				ranges.push_back(ranges_[argtens]);
			}

			// func range
			ranges_.emplace(func, generate_range(func, ranges));
		}
	}

	std::unordered_map<teq::iTensor*,estd::NumRange<T>> ranges_;
};

}

#endif // ETEQ_STABILIZER_HPP
