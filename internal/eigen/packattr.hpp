
#ifndef EIGEN_PACKATTR_HPP
#define EIGEN_PACKATTR_HPP

#include "internal/teq/teq.hpp"
#include "internal/eigen/generated/dtype.hpp"
#include "internal/eigen/convert.hpp"

namespace eigen
{

const std::string no_argument_err = "cannot operate without inputs";

using DTypesT = std::vector<egen::_GENERATED_DTYPE>;

template <typename T>
using PairVecT = std::vector<std::pair<T,T>>;

using OptDimsT = std::optional<teq::DimsT>;

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
		global::fatalf("cannot decode odd vector %s into vec of pairs",
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
		global::fatal("unknown attribute");
	}

	void unpack (T& out, const marsh::iAttributed& attrib) const
	{
		global::fatal("unknown attribute");
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
		global::fatalf("cannot find `%s` attribute", key.c_str());
	}
	return attr;
}

template <>
struct Packer<egen::_GENERATED_DTYPE>
{
	static std::string key_;

	std::string get_key (void) const
	{
		return key_;
	}

	void pack (marsh::iAttributed& attrib, egen::_GENERATED_DTYPE dtye) const
	{
		attrib.add_attr(key_,
			std::make_unique<marsh::String>(egen::name_type(dtye)));
	}

	void unpack (egen::_GENERATED_DTYPE& out, const marsh::iAttributed& attrib) const
	{
		auto attr = get_attr(*this, attrib);
		auto& sarr = static_cast<const marsh::String&>(*attr);
		out = egen::get_type(sarr.to_string());
	}
};

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
		attrib.add_attr(key_, std::make_unique<marsh::NumArray<int64_t>>(
			encode_pair(dims)));
	}

	void unpack (PairVecT<teq::DimT>& out, const marsh::iAttributed& attrib) const
	{
		auto attr = get_attr(*this, attrib);
		auto& narr = static_cast<const marsh::NumArray<int64_t>&>(*attr);
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
		if (std::any_of(ranks.begin(), ranks.end(),
			[](std::pair<teq::RankT,teq::RankT> rpair)
			{
				return rpair.first >= teq::rank_cap ||
					rpair.second >= teq::rank_cap;
			}))
		{
			global::fatalf("cannot reference ranks beyond rank_cap %d: %s",
				teq::rank_cap, to_string(ranks).c_str());
		}
		attrib.add_attr(key_, std::make_unique<marsh::NumArray<int64_t>>(
			encode_pair(ranks)));
	}

	void unpack (PairVecT<teq::RankT>& out, const marsh::iAttributed& attrib) const
	{
		auto attr = get_attr(*this, attrib);
		auto& narr = static_cast<const marsh::NumArray<int64_t>&>(*attr);
		auto& encoding = narr.contents_;
		out = decode_pair<teq::RankT>(encoding);
	}
};

template <>
struct Packer<teq::DimsT>
{
	static std::string key_;

	std::string get_key (void) const
	{
		return key_;
	}

	void pack (marsh::iAttributed& attrib, teq::DimsT dims) const
	{
		std::vector<int64_t> idims(dims.begin(), dims.end());
		attrib.add_attr(key_,
			std::make_unique<marsh::NumArray<int64_t>>(idims));
	}

	void unpack (teq::DimsT& out, const marsh::iAttributed& attrib) const
	{
		auto attr = get_attr(*this, attrib);
		auto& narr = static_cast<const marsh::NumArray<int64_t>&>(*attr);
		auto& encoding = narr.contents_;
		out = teq::DimsT(encoding.begin(), encoding.end());
	}
};

template <>
struct Packer<teq::RanksT>
{
	static std::string key_;

	std::string get_key (void) const
	{
		return key_;
	}

	void pack (marsh::iAttributed& attrib, teq::RanksT ranks) const
	{
		if (std::any_of(ranks.begin(), ranks.end(),
			[](teq::RankT rank)
			{
				return rank >= teq::rank_cap;
			}))
		{
			global::fatalf("cannot reference ranks beyond rank_cap %d: %s",
				teq::rank_cap, fmts::to_string(
					ranks.begin(), ranks.end()).c_str());
		}
		std::vector<int64_t> sranks(ranks.begin(), ranks.end());
		attrib.add_attr(key_,
			std::make_unique<marsh::NumArray<int64_t>>(sranks));
	}

	void unpack (teq::RanksT& out, const marsh::iAttributed& attrib) const
	{
		auto attr = get_attr(*this, attrib);
		auto& narr = static_cast<const marsh::NumArray<int64_t>&>(*attr);
		auto& encoding = narr.contents_;
		out = teq::RanksT(encoding.begin(), encoding.end());
	}
};

template <>
struct Packer<StorageIndicesT>
{
	static std::string key_;

	std::string get_key (void) const
	{
		return key_;
	}

	void pack (marsh::iAttributed& attrib, const StorageIndicesT& dims) const
	{
		std::vector<int64_t> idims(dims.begin(), dims.end());
		attrib.add_attr(key_,
			std::make_unique<marsh::NumArray<int64_t>>(idims));
	}

	void unpack (StorageIndicesT& out, const marsh::iAttributed& attrib) const
	{
		auto attr = get_attr(*this, attrib);
		auto& narr = static_cast<const marsh::NumArray<int64_t>&>(*attr);
		auto& encoding = narr.contents_;
		out = StorageIndicesT(encoding.begin(), encoding.end());
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
			global::fatalf("cannot specify %d ranks when %d (rank_cap) "
				"are available", n, teq::rank_cap);
		}
		std::vector<int64_t> sranks(ranks.begin(), ranks.end());
		std::sort(sranks.begin(), sranks.end());
		if (std::any_of(sranks.begin(), sranks.end(),
			[](teq::RankT rank)
			{
				return rank > teq::rank_cap;
			}))
		{
			global::fatalf("cannot reference ranks beyond rank_cap %d: %s",
				teq::rank_cap, fmts::to_string(
					sranks.begin(), sranks.end()).c_str());
		}
		attrib.add_attr(key_,
			std::make_unique<marsh::NumArray<int64_t>>(sranks));
	}

	void unpack (std::set<teq::RankT>& out, const marsh::iAttributed& attrib) const
	{
		auto attr = get_attr(*this, attrib);
		auto& narr = static_cast<const marsh::NumArray<int64_t>&>(*attr);
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
		auto& narr = static_cast<const marsh::Number<int64_t>&>(*attr);
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
		auto& narr = static_cast<const marsh::NumArray<int64_t>&>(*attr);
		auto& encoding = narr.contents_;
		teq::DimsT slist(encoding.begin(), encoding.end());
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
		auto& tarr = static_cast<const teq::TensorObj&>(*attr);
		out = tarr.get_tensor();
	}
};

void pack_attr (marsh::iAttributed& attrib);

struct AttrKeyVal final
{
	std::string key_;

	std::string val_;
};

template <typename ...ARGS>
void pack_attr (marsh::iAttributed& attrib, const AttrKeyVal& attr_val, ARGS... args)
{
	attrib.add_attr(attr_val.key_,
		std::make_unique<marsh::String>(attr_val.val_));
	pack_attr(attrib, args...);
}

template <typename T, typename ...ARGS>
void pack_attr (marsh::iAttributed& attrib, T attr_val, ARGS... args)
{
	Packer<T>().pack(attrib, attr_val);
	pack_attr(attrib, args...);
}

OptDimsT unpack_extend (teq::Shape inshape, const marsh::iAttributed& attrib);

}

#endif // EIGEN_PACKATTR_HPP
