#include "marsh/objs.hpp"

#include "teq/teq.hpp"

#ifndef EIGEN_PACKATTR_HPP
#define EIGEN_PACKATTR_HPP

namespace eigen
{

const std::string commutative_attr = "commutative";

template <typename T>
using PairVecT = std::vector<std::pair<T,T>>;

template <typename T>
std::string to_string (const PairVecT<T>& pairs)
{
	PairVecT<int> readable_pairs(pairs.begin(), pairs.end());
	return fmts::to_string(readable_pairs.begin(), readable_pairs.end());
}

template <typename T>
std::vector<int64_t> encode_pair (const PairVecT<T>& pairs)
{
	size_t npairs = pairs.size();
	std::vector<int64_t> out;
	out.reserve(npairs * 2);
	for (auto& p : pairs)
	{
		out.push_back(p.first);
		out.push_back(p.second);
	}
	return out;
}

template <typename T>
const PairVecT<T> decode_pair (const std::vector<int64_t>& encoding)
{
	PairVecT<T> out;
	size_t n = encoding.size();
	if (1 == n % 2)
	{
		logs::fatalf("cannot decode odd vector %s into vec of pairs",
			fmts::to_string(encoding.begin(), encoding.end()).c_str());
	}
	out.reserve(n / 2);
	for (size_t i = 0; i < n; i += 2)
	{
		out.push_back({encoding[i], encoding[i + 1]});
	}
	return out;
}

template <typename T>
struct Packer final
{
	std::string get_key (void) const
	{
		return "";
	}

	void pack (marsh::iAttributed& attrib, T pack) const
	{
		logs::fatal("unknown attribute");
	}

	void unpack (T& out, const marsh::iAttributed& attrib) const
	{
		logs::fatal("unknown attribute");
	}
};

template <typename T>
const marsh::iObject* get_attr (const Packer<T>& packer,
	const marsh::iAttributed& attrib)
{
	auto key = packer.get_key();
	auto attr = attrib.get_attr(key);
	if (nullptr == attr)
	{
		logs::fatalf("cannot find %s attribute", key.c_str());
	}
	return attr;
}

template <>
struct Packer<PairVecT<teq::DimT>>
{
	static std::string key_;

	std::string get_key (void) const
	{
		return key_;
	}

	void pack (marsh::iAttributed& attrib, PairVecT<teq::DimT> dims) const
	{
		size_t n = dims.size();
		if (n > teq::rank_cap)
		{
			logs::fatalf("cannot specify %d dimensions when %d (rank_cap) "
				"are available", n, teq::rank_cap);
		}
		if (n == 0)
		{
			logs::fatal("cannot find dimensions");
		}
		attrib.add_attr(key_, std::make_unique<marsh::NumArray<int64_t>>(
			encode_pair(dims)));
	}

	void unpack (PairVecT<teq::DimT>& out, const marsh::iAttributed& attrib) const
	{
		auto attr = get_attr(*this, attrib);
		auto& narr = dynamic_cast<const marsh::NumArray<int64_t>&>(*attr);
		auto& encoding = narr.contents_;
		out = decode_pair<teq::DimT>(encoding);
	}
};

template <>
struct Packer<PairVecT<teq::RankT>>
{
	static std::string key_;

	std::string get_key (void) const
	{
		return key_;
	}

	void pack (marsh::iAttributed& attrib, PairVecT<teq::RankT> ranks) const
	{
		size_t n = ranks.size();
		if (n > teq::rank_cap)
		{
			logs::fatalf("cannot specify %d ranks when %d (rank_cap) "
				"are available", n, teq::rank_cap);
		}
		if (n == 0)
		{
			logs::fatal("cannot find pair of ranks");
		}
		if (std::any_of(ranks.begin(), ranks.end(),
			[](std::pair<teq::RankT,teq::RankT> rpair)
			{
				return rpair.first >= teq::rank_cap ||
					rpair.second >= teq::rank_cap;
			}))
		{
			logs::fatalf("cannot reference ranks beyond rank_cap %d: %s",
				teq::rank_cap, to_string(ranks).c_str());
		}
		attrib.add_attr(key_, std::make_unique<marsh::NumArray<int64_t>>(
			encode_pair(ranks)));
	}

	void unpack (PairVecT<teq::RankT>& out, const marsh::iAttributed& attrib) const
	{
		auto attr = get_attr(*this, attrib);
		auto& narr = dynamic_cast<const marsh::NumArray<int64_t>&>(*attr);
		auto& encoding = narr.contents_;
		out = decode_pair<teq::RankT>(encoding);
	}
};

template <>
struct Packer<std::vector<teq::DimT>>
{
	static std::string key_;

	std::string get_key (void) const
	{
		return key_;
	}

	void pack (marsh::iAttributed& attrib, std::vector<teq::DimT> dims) const
	{
		size_t n = dims.size();
		if (n > teq::rank_cap)
		{
			logs::fatalf("cannot specify %d dimensions when %d (rank_cap) "
				"are available", n, teq::rank_cap);
		}
		if (n == 0)
		{
			logs::fatal("cannot find dimensions");
		}
		std::vector<int64_t> idims(dims.begin(), dims.end());
		attrib.add_attr(key_,
			std::make_unique<marsh::NumArray<int64_t>>(idims));
	}

	void unpack (std::vector<teq::DimT>& out, const marsh::iAttributed& attrib) const
	{
		auto attr = get_attr(*this, attrib);
		auto& narr = dynamic_cast<const marsh::NumArray<int64_t>&>(*attr);
		auto& encoding = narr.contents_;
		out = std::vector<teq::DimT>(encoding.begin(), encoding.end());
	}
};

template <>
struct Packer<std::vector<teq::RankT>>
{
	static std::string key_;

	std::string get_key (void) const
	{
		return key_;
	}

	void pack (marsh::iAttributed& attrib, std::vector<teq::RankT> ranks) const
	{
		size_t n = ranks.size();
		if (n > teq::rank_cap)
		{
			logs::fatalf("cannot specify %d ranks when %d (rank_cap) "
				"are available", n, teq::rank_cap);
		}
		if (n == 0)
		{
			logs::fatal("cannot find ranks");
		}
		if (std::any_of(ranks.begin(), ranks.end(),
			[](teq::RankT rank)
			{
				return rank >= teq::rank_cap;
			}))
		{
			logs::fatalf("cannot reference ranks beyond rank_cap %d: %s",
				teq::rank_cap, fmts::to_string(
					ranks.begin(), ranks.end()).c_str());
		}
		std::vector<int64_t> sranks(ranks.begin(), ranks.end());
		attrib.add_attr(key_,
			std::make_unique<marsh::NumArray<int64_t>>(sranks));
	}

	void unpack (std::vector<teq::RankT>& out, const marsh::iAttributed& attrib) const
	{
		auto attr = get_attr(*this, attrib);
		auto& narr = dynamic_cast<const marsh::NumArray<int64_t>&>(*attr);
		auto& encoding = narr.contents_;
		out = std::vector<teq::RankT>(encoding.begin(), encoding.end());
	}
};

template <>
struct Packer<std::set<teq::RankT>>
{
	static std::string key_;

	std::string get_key (void) const
	{
		return key_;
	}

	void pack (marsh::iAttributed& attrib, std::set<teq::RankT> ranks) const
	{
		size_t n = ranks.size();
		if (n > teq::rank_cap)
		{
			logs::fatalf("cannot specify %d ranks when %d (rank_cap) "
				"are available", n, teq::rank_cap);
		}
		std::vector<int64_t> sranks(ranks.begin(), ranks.end());
		std::sort(sranks.begin(), sranks.end());
		attrib.add_attr(key_,
			std::make_unique<marsh::NumArray<int64_t>>(sranks));
	}

	void unpack (std::set<teq::RankT>& out, const marsh::iAttributed& attrib) const
	{
		auto attr = get_attr(*this, attrib);
		auto& narr = dynamic_cast<const marsh::NumArray<int64_t>&>(*attr);
		auto& encoding = narr.contents_;
		out = std::set<teq::RankT>(encoding.begin(), encoding.end());
	}
};

template <>
struct Packer<teq::RankT>
{
	static std::string key_;

	std::string get_key (void) const
	{
		return key_;
	}

	void pack (marsh::iAttributed& attrib, teq::RankT rank) const
	{
		attrib.add_attr(key_, std::make_unique<marsh::Number<int64_t>>(rank));
	}

	void unpack (teq::RankT& out, const marsh::iAttributed& attrib) const
	{
		auto attr = get_attr(*this, attrib);
		auto& narr = dynamic_cast<const marsh::Number<int64_t>&>(*attr);
		out = narr.val_;
	}
};

template <>
struct Packer<teq::Shape>
{
	static std::string key_;

	std::string get_key (void) const
	{
		return key_;
	}

	void pack (marsh::iAttributed& attrib, teq::Shape shape) const
	{
		std::vector<int64_t> slist(shape.begin(), shape.end());
		attrib.add_attr(key_,
			std::make_unique<marsh::NumArray<int64_t>>(slist));
	}

	void unpack (teq::Shape& out, const marsh::iAttributed& attrib) const
	{
		auto attr = get_attr(*this, attrib);
		auto& narr = dynamic_cast<const marsh::NumArray<int64_t>&>(*attr);
		auto& encoding = narr.contents_;
		std::vector<teq::DimT> slist(encoding.begin(), encoding.end());
		out = teq::Shape(slist);
	}
};

template <>
struct Packer<teq::TensptrT>
{
	static std::string key_;

	std::string get_key (void) const
	{
		return key_;
	}

	void pack (marsh::iAttributed& attrib, teq::TensptrT tens) const
	{
		attrib.add_attr(key_,
			std::make_unique<teq::TensorObj>(tens));
	}

	void unpack (teq::TensptrT& out, const marsh::iAttributed& attrib) const
	{
		auto attr = get_attr(*this, attrib);
		auto& tarr = dynamic_cast<const teq::TensorObj&>(*attr);
		out = tarr.get_tensor();
	}
};

void pack_attr (marsh::iAttributed& attrib);

template <typename T, typename ...ARGS>
void pack_attr (marsh::iAttributed& attrib, T attr_val, ARGS... args)
{
	Packer<T>().pack(attrib, attr_val);
	pack_attr(attrib, args...);
}

std::vector<teq::DimT> unpack_extend (
	teq::Shape inshape, const marsh::iAttributed& attrib);

}

#endif // EIGEN_PACKATTR_HPP
