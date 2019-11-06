#include "teq/traveler.hpp"

#include "eigen/generated/opcode.hpp"
#include "eigen/generated/dtype.hpp"

#ifndef EIGEN_STABILIZER_HPP
#define EIGEN_STABILIZER_HPP

namespace eigen // todo: move to its own module
{

static const double pi = std::acos(-(double) 1);

template <typename T>
using NumRangesT = std::vector<estd::NumRange<T>>;

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

template <typename T, typename std::enable_if<
    std::is_integral<T>::value>::type* = nullptr>
estd::NumRange<T> pow_range (const NumRangesT<T>& ranges)
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

template <typename T, typename std::enable_if<
    !std::is_integral<T>::value>::type* = nullptr>
estd::NumRange<T> pow_range (const NumRangesT<T>& ranges)
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
		// look for nearest even for highest
		size_t ilexp = std::ceil(lower_exp);
		upper = std::max(upper, std::pow(lower,
			ilexp % 2 ? ilexp + 1 : ilexp));
		// exponent be a decimal and base can be negative
		lower = std::nan("");
		lower_exp = std::ceil(lower_exp);
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

template <typename T>
estd::NumRange<T> generate_range (teq::iFunctor* func, const NumRangesT<T>& ranges)
{
	teq::Opcode opcode = func->get_opcode();
	estd::NumRange<T> outrange;
	switch (opcode.code_)
	{
		case egen::ABS:
		{
			T lower = ranges[0].lower_;
			T upper = ranges[0].upper_;
			std::vector<T> bounds = {std::abs(lower), std::abs(upper)};
			if (lower < 0 && upper > 0)
			{
				bounds.push_back(0);
			}
			outrange = estd::NumRange<T>(
				*std::min_element(bounds.begin(), bounds.end()),
				*std::max_element(bounds.begin(), bounds.end()));
		}
			break;
		case egen::NEG:
			outrange = estd::NumRange<T>(-ranges[0].lower_, -ranges[0].upper_);
			break;
		case egen::SIN:
		{
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
					outrange = estd::NumRange<T>(sl, su);
				}
				else
				{
					// cos(ql) is the nearest range extrema above lower
					// cos(qu) is the nearest range extrema below upper
					std::vector<T> bounds = {std::cos(ql), std::cos(qu), sl, su};
					outrange = estd::NumRange<T>(
						*std::min_element(bounds.begin(), bounds.end()),
						*std::max_element(bounds.begin(), bounds.end()));
				}
			}
			else
			{
				outrange = estd::NumRange<T>(-1, 1);
			}
		}
			break;
		case egen::COS:
		{
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
					outrange = estd::NumRange<T>(sl, su);
				}
				else
				{
					// cos(ql) is the nearest range extrema above lower
					// cos(qu) is the nearest range extrema below upper
					std::vector<T> bounds = {std::cos(ql), std::cos(qu), sl, su};
					outrange = estd::NumRange<T>(
						*std::min_element(bounds.begin(), bounds.end()),
						*std::max_element(bounds.begin(), bounds.end()));
				}
			}
			else
			{
				outrange = estd::NumRange<T>(-1, 1);
			}
		}
			break;
		case egen::TAN:
		{
			T lower = ranges[0].lower_;
			T upper = ranges[0].upper_;

			if (std::round(lower / pi) * pi == std::round(upper / pi) * pi)
			{
				// ranges are within finite values
				outrange = estd::NumRange<T>(std::tanh(lower), std::tanh(upper));
			}
			else
			{
				T inf = std::numeric_limits<T>::infinity();
				outrange = estd::NumRange<T>(-inf, inf);
			}
		}
			break;
		case egen::EXP:
			outrange = estd::NumRange<T>(std::exp(ranges[0].lower_), std::exp(ranges[0].upper_));
			break;
		case egen::LOG:
			outrange = estd::NumRange<T>(std::log(ranges[0].lower_), std::log(ranges[0].upper_));
			break;
		case egen::SQRT:
			outrange = estd::NumRange<T>(std::sqrt(ranges[0].lower_), std::sqrt(ranges[0].upper_));
			break;
		case egen::ROUND:
			outrange = estd::NumRange<T>(std::round(ranges[0].lower_), std::round(ranges[0].upper_));
			break;
		case egen::SQUARE:
		{
			T lower = ranges[0].lower_;
			T upper = ranges[0].upper_;
			outrange = estd::NumRange<T>(
				lower < 0 && upper > 0 ? 0 : lower * lower,
				upper * upper);
		}
			break;
		case egen::CUBE:
			outrange = estd::NumRange<T>(cube(ranges[0].lower_), cube(ranges[0].upper_));
			break;
		case egen::SIGMOID:
			outrange = estd::NumRange<T>(sigmoid(ranges[0].lower_), sigmoid(ranges[0].upper_));
			break;
		case egen::TANH:
			outrange = estd::NumRange<T>(tanh(ranges[0].lower_), tanh(ranges[0].upper_));
			break;
		case egen::PERMUTE:
		case egen::EXTEND:
		case egen::RESHAPE:
		case egen::REVERSE:
		case egen::RAND_UNIF:
		case egen::SLICE:
		case egen::STRIDE:
			outrange = ranges[0];
			break;
		case egen::PAD:
		case egen::SCATTER:
		{
			std::vector<T> bounds = {ranges[0].lower_, ranges[0].upper_, 0};
			outrange = estd::NumRange<T>(
				*std::min_element(bounds.begin(), bounds.end()),
				*std::max_element(bounds.begin(), bounds.end()));
		}
			break;
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
			outrange = estd::NumRange<T>(0, teq::rank_cap == return_dim ?
				shape.n_elems() : shape.at(return_dim));
		}
			break;
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
			outrange = estd::NumRange<T>(
				*std::min_element(lbounds.begin(), lbounds.end()),
				*std::max_element(ubounds.begin(), ubounds.end()));
		}
			break;
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
			outrange = estd::NumRange<T>(
				*std::min_element(lbounds.begin(), lbounds.end()),
				*std::max_element(ubounds.begin(), ubounds.end()));
		}
			break;
		case egen::POW:
			outrange = pow_range(ranges);
			break;
		case egen::ADD:
			outrange = estd::NumRange<T>(
				ranges[0].lower_ + ranges[1].lower_,
				ranges[0].upper_ + ranges[1].upper_);
			break;
		case egen::SUB:
			outrange = estd::NumRange<T>(
				ranges[0].lower_ + ranges[1].upper_,
				ranges[0].upper_ + ranges[1].lower_);
			break;
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
			outrange = estd::NumRange<T>(
				*std::min_element(bounds.begin(), bounds.end()),
				*std::max_element(bounds.begin(), bounds.end()));
		}
			break;
		case egen::DIV:
		{
			T alower = ranges[0].lower_;
			T aupper = ranges[0].upper_;
			T blower = ranges[1].lower_;
			T bupper = ranges[1].upper_;
			std::vector<T> bounds = {
				alower / blower,
				alower / bupper,
				aupper / blower,
				aupper / bupper};
			outrange = estd::NumRange<T>(
				*std::min_element(bounds.begin(), bounds.end()),
				*std::max_element(bounds.begin(), bounds.end()));
		}
			break;
		case egen::MIN:
			outrange = estd::NumRange<T>(
				std::min(ranges[0].lower_,ranges[1].lower_),
				std::min(ranges[0].upper_,ranges[1].upper_));
			break;
		case egen::MAX:
			outrange = estd::NumRange<T>(
				std::max(ranges[0].lower_,ranges[1].lower_),
				std::max(ranges[0].upper_,ranges[1].upper_));
			break;
		case egen::EQ:
		case egen::NEQ:
		case egen::LT:
		case egen::GT:
			outrange = estd::NumRange<T>((T) 0, (T) 1);
			break;
		case egen::REDUCE_SUM:
		{
			auto& arg = func->get_children()[0];
			auto coorder = arg.get_coorder();
			assert(coorder != nullptr);
			const teq::Shape& shape = arg.get_tensor()->shape();
			teq::NElemT nreds = 1;
			bool reduced = false;
			coorder->access(
				[&](const teq::MatrixT& args)
				{
					for (teq::RankT i = 0; i < teq::rank_cap; ++i)
					{
						auto d = args[0][i];
						if (d < teq::rank_cap)
						{
							reduced = true;
							nreds *= shape.at(d);
						}
					}
				});
			if (false == reduced)
			{
				reduced = shape.n_elems();
			}
			outrange = estd::NumRange<T>(
				ranges[0].lower_ * nreds,
				ranges[0].upper_ * nreds);
		}
			break;
		case egen::REDUCE_PROD:
		{
			auto& arg = func->get_children()[0];
			auto coorder = arg.get_coorder();
			assert(coorder != nullptr);
			const teq::Shape& shape = arg.get_tensor()->shape();
			teq::NElemT nreds = 1;
			bool reduced = false;
			coorder->access(
				[&](const teq::MatrixT& args)
				{
					for (teq::RankT i = 0; i < teq::rank_cap; ++i)
					{
						auto d = args[0][i];
						if (d < teq::rank_cap)
						{
							reduced = true;
							nreds *= shape.at(d);
						}
					}
				});
			if (false == reduced)
			{
				reduced = shape.n_elems();
			}
			outrange = estd::NumRange<T>(
				std::pow(ranges[0].lower_, nreds),
				std::pow(ranges[0].upper_, nreds));
		}
			break;
		case egen::REDUCE_MIN:
		case egen::REDUCE_MAX:
			outrange = ranges[0];
			break;
		case egen::MATMUL:
		{
			// matmul = <left> * <right> then reduce sum by common dimensions
			// so apply range rule for product, then for reduce sum
			auto& arg = func->get_children()[0];
			auto coorder = arg.get_coorder();
			const teq::Shape& shape = arg.get_tensor()->shape();
			assert(nullptr != coorder);
			teq::NElemT ncommons = 1;
			coorder->access(
				[&](const teq::MatrixT& args)
				{
					for (teq::RankT i = 0; i < teq::rank_cap &&
						args[0][i] < teq::rank_cap; ++i)
					{
						ncommons *= shape.at(args[0][i]);
					}
				});
			outrange = estd::NumRange<T>(
				ranges[0].lower_ * ranges[1].lower_ * ncommons,
				ranges[0].upper_ * ranges[1].upper_ * ncommons);
		}
			break;
		case egen::CONV:
		{
			// conv = <image> * <kernel> then reduce by kernel dimensions that convolves
			// apply range rule similar to matmul
			auto& arg = func->get_children()[1];
			const teq::Shape& shape = arg.get_tensor()->shape();
			auto coorder = arg.get_coorder();
			assert(nullptr != coorder);
			teq::NElemT nkern = 1;
			coorder->access(
				[&](const teq::MatrixT& args)
				{
					for (teq::RankT i = 0; i < teq::rank_cap &&
						args[0][i] < teq::rank_cap; ++i)
					{
						nkern *= shape.at(i);
					}
				});
			outrange = estd::NumRange<T>(
				ranges[0].lower_ * ranges[1].lower_ * nkern,
				ranges[0].upper_ * ranges[1].upper_ * nkern);
		}
			break;
		default:
			logs::fatalf("Unknown op %s", opcode.name_.c_str());
	}
	return outrange;
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
			if (egen::get_type<T>() == leaf->type_code() && leaf->is_const())
			{
				auto data = (T*) leaf->data();
				teq::NElemT n = leaf->shape().n_elems();
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
			NumRangesT<T> ranges;
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

#endif // EIGEN_STABILIZER_HPP
