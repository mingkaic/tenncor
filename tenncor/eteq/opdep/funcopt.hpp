///
/// funcopt.hpp
/// eteq
///
/// Purpose:
/// Check if operator represented by opcode is redundant
/// based on attribute and input shapes
///

#ifndef ETEQ_FUNCOPT_HPP
#define ETEQ_FUNCOPT_HPP

#include "internal/eigen/eigen.hpp"

namespace eteq
{

// todo: move these to eigen and auto-generate
template <egen::_GENERATED_OPCODE OPCODE>
struct FuncOpt final
{
	/// Return true if functor is redunant provided attrs and shapes
	template <typename T>
	bool is_redundant (const marsh::Maps& attrs, const teq::TensptrsT& args)
	{
		return false;
	}
};

template <>
struct FuncOpt<egen::CAST>
{
	template <typename T>
	bool is_redundant (const marsh::Maps& attrs, const teq::TensptrsT& args)
	{
		auto argtype = (egen::_GENERATED_DTYPE)
			args.front()->get_meta().type_code();
		bool redundant = argtype == egen::get_type<T>();
		if (redundant)
		{
			global::debugf("redundantly casting to same type %s",
				egen::name_type(argtype).c_str());
		}
		return redundant;
	}
};

struct NnaryOpt
{
	virtual ~NnaryOpt (void) = default;

	template <typename T>
	bool is_redundant (const marsh::Maps& attrs, const teq::TensptrsT& args)
	{
		bool redundant = args.size() < 2;
		if (redundant)
		{
			// assuming empty args is handled before
			global::debug("redundantly performing nnary op on a single arg... "
				"treating as identity");
		}
		return redundant;
	}
};

template <>
struct FuncOpt<egen::ADD> final : private NnaryOpt
{
	using NnaryOpt::is_redundant;
};

template <>
struct FuncOpt<egen::MUL> final : private NnaryOpt
{
	using NnaryOpt::is_redundant;
};

struct ReduceOpt
{
	virtual ~ReduceOpt (void) = default;

	template <typename T>
	bool is_redundant (const marsh::Maps& attrs, const teq::TensptrsT& args)
	{
		std::set<teq::RankT> ranks;
		eigen::Packer<std::set<teq::RankT>>().unpack(ranks, attrs);
		bool redundant = ranks.empty();
		if (redundant)
		{
			global::debugf("reducing with no significant dimensions... "
				"treating as identity: (dims=%s, shape=%s)",
				fmts::to_string(ranks.begin(), ranks.end()).c_str(),
				args.front()->shape().to_string().c_str());
		}
		return redundant;
	}
};

template <>
struct FuncOpt<egen::REDUCE_SUM> final : private ReduceOpt
{
	using ReduceOpt::is_redundant;
};

template <>
struct FuncOpt<egen::REDUCE_PROD> final : private ReduceOpt
{
	using ReduceOpt::is_redundant;
};

template <>
struct FuncOpt<egen::REDUCE_MIN> final : private ReduceOpt
{
	using ReduceOpt::is_redundant;
};

template <>
struct FuncOpt<egen::REDUCE_MAX> final : private ReduceOpt
{
	using ReduceOpt::is_redundant;
};

template <>
struct FuncOpt<egen::ARGMAX> final
{
	template <typename T>
	bool is_redundant (const marsh::Maps& attrs, const teq::TensptrsT& args)
	{
		teq::RankT return_dim;
		eigen::Packer<teq::RankT>().unpack(return_dim, attrs);
		teq::Shape shape = args.front()->shape();
		bool redundant = return_dim < teq::rank_cap && shape.at(return_dim) == 1;
		if (redundant)
		{
			global::debugf("argreducing with no significant dimensions... "
				"treating as identity: (return_dim=%d, shape=%s)",
				(int) return_dim, shape.to_string().c_str());
		}
		return redundant;
	}
};

template <>
struct FuncOpt<egen::SLICE> final
{
	template <typename T>
	bool is_redundant (const marsh::Maps& attrs, const teq::TensptrsT& args)
	{
		eigen::PairVecT<teq::DimT> extents;
		eigen::Packer<eigen::PairVecT<teq::DimT>>().unpack(extents, attrs);
		teq::Shape shape = args.front()->shape();
		bool redundant = true;
		for (size_t i = 0, n = std::min(extents.size(),
			(size_t) teq::rank_cap); i < n && redundant; ++i)
		{
			auto& exts = extents[i];
			if (exts.second == 0)
			{
				global::fatalf("cannot create slice with 0 dimension at "
					"index %d (extents=%s)", i,
					eigen::to_string(extents).c_str());
			}
			redundant = redundant && exts.first == 0 &&
				exts.second > shape.at(i);
		}
		if (redundant)
		{
			global::debugf("slice parameter covers whole tensor... "
				"treating as identity: (extents=%s)",
				eigen::to_string(extents).c_str());
		}
		return redundant;
	}
};

template <>
struct FuncOpt<egen::PAD> final
{
	template <typename T>
	bool is_redundant (const marsh::Maps& attrs, const teq::TensptrsT& args)
	{
		eigen::PairVecT<teq::DimT> paddings;
		eigen::Packer<eigen::PairVecT<teq::DimT>>().unpack(paddings, attrs);
		bool redundant = std::all_of(paddings.begin(), paddings.end(),
			[](std::pair<teq::DimT,teq::DimT> pad)
			{
				return pad.first == 0 && pad.second == 0;
			});
		if (redundant)
		{
			global::debugf("padding are all zero... "
				"treating as identity: (paddings=%s)",
				eigen::to_string(paddings).c_str());
		}
		return redundant;
	}
};

template <>
struct FuncOpt<egen::SCATTER> final
{
	template <typename T>
	bool is_redundant (const marsh::Maps& attrs, const teq::TensptrsT& args)
	{
		teq::Shape outshape;
		eigen::Packer<teq::Shape>().unpack(outshape, attrs);
		bool redundant = args.front()->shape().compatible_after(outshape, 0);
		if (redundant)
		{
			global::debugf("scattering produces the same shape %s",
				outshape.to_string().c_str());
		}
		return redundant;
	}
};

template <>
struct FuncOpt<egen::PERMUTE> final
{
	template <typename T>
	bool is_redundant (const marsh::Maps& attrs, const teq::TensptrsT& args)
	{
		std::vector<teq::RankT> order;
		eigen::Packer<std::vector<teq::RankT>>().unpack(order, attrs);
		bool redundant = order.empty() ? true : (order[0] == 0);
		for (size_t i = 1, n = std::min(order.size(), (size_t) teq::rank_cap);
			i < n && redundant; ++i)
		{
			redundant = redundant && (order[i] == (order[i-1] + 1));
		}
		if (redundant)
		{
			global::debug("permuting with same "
				"dimensions ... treating as identity");
		}
		return redundant;
	}
};

template <>
struct FuncOpt<egen::EXTEND> final
{
	template <typename T>
	bool is_redundant (const marsh::Maps& attrs, const teq::TensptrsT& args)
	{
		std::vector<teq::DimT> bcast = eigen::unpack_extend(
			args.front()->shape(), attrs);
		bool redundant = nullptr != attrs.get_attr(
			eigen::Packer<std::vector<teq::DimT>>().get_key()) &&
			(bcast.empty() || std::all_of(bcast.begin(), bcast.end(),
			[](teq::DimT d) { return 1 == d; }));
		if (redundant)
		{
			global::debug("extending with nothing... treating as identity");
		}
		return redundant;
	}
};

template <>
struct FuncOpt<egen::CONCAT> final
{
	template <typename T>
	bool is_redundant (const marsh::Maps& attrs, const teq::TensptrsT& args)
	{
		bool redundant = args.size() == 1;
		if (redundant)
		{
			global::debug("concatenating a single node... treating as identity");
		}
		return redundant;
	}
};

template <>
struct FuncOpt<egen::RESHAPE> final
{
	template <typename T>
	bool is_redundant (const marsh::Maps& attrs, const teq::TensptrsT& args)
	{
		teq::Shape outshape;
		eigen::Packer<teq::Shape>().unpack(outshape, attrs);
		bool redundant = outshape.compatible_after(args.front()->shape(), 0);
		if (redundant)
		{
			global::debugf("outshape of reshape is the same shape as inshape "
				"%s... treating as identity", outshape.to_string().c_str());
		}
		return redundant;
	}
};

}

#endif // ETEQ_FUNCOPT_HPP
