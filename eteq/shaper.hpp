#include <optional>

#include "marsh/attrs.hpp"

#include "teq/shape.hpp"

#include "eigen/generated/opcode.hpp"

#ifndef ETEQ_SHAPER_HPP
#define ETEQ_SHAPER_HPP

namespace eteq
{

using ShapeOpt = std::optional<teq::ShapeSignature>;

using ShapesT = std::vector<teq::ShapeSignature>;

// todo: move these to eigen and auto-generate
template <egen::_GENERATED_OPCODE OPCODE>
struct ShapeParser final
{
	/// Return output shape if operator is not redundant
	/// given specified attributes and input shapes
	ShapeOpt shape (const marsh::Maps& attrs, const ShapesT& shapes)
	{
		return shapes.front();
	}
};

struct ReducePacker
{
	virtual ~ReducePacker (void) = default;

	ShapeOpt shape (const marsh::Maps& attrs, const ShapesT& shapes)
	{
		std::set<teq::RankT> ranks;
		eigen::Packer<std::set<teq::RankT>>().unpack(ranks, attrs);
		std::string rank_snapshot = fmts::to_string(
			ranks.begin(), ranks.end());

		teq::ShapeSignature shape = shapes.front();
		std::vector<teq::DimT> slist(shape.begin(), shape.end());
		for (auto it = ranks.begin(), et = ranks.end(); it != et;)
		{
			if (slist.at(*it) != 1)
			{
				slist[*it] = 1;
				++it;
			}
			else
			{
				it = ranks.erase(it);
			}
		}
		if (ranks.empty())
		{
			logs::debugf("reducing with no significant dimensions... "
				"treating as identity: (dims=%s, shape=%s)",
				rank_snapshot.c_str(),
				shape.to_string().c_str());
			return ShapeOpt();
		}
		return teq::ShapeSignature(slist);
	}
};

template <>
struct ShapeParser<egen::REDUCE_SUM> final : private ReducePacker
{
	using ReducePacker::shape;
};

template <>
struct ShapeParser<egen::REDUCE_PROD> final : private ReducePacker
{
	using ReducePacker::shape;
};

template <>
struct ShapeParser<egen::REDUCE_MIN> final : private ReducePacker
{
	using ReducePacker::shape;
};

template <>
struct ShapeParser<egen::REDUCE_MAX> final : private ReducePacker
{
	using ReducePacker::shape;
};

template <>
struct ShapeParser<egen::ARGMAX> final
{
	ShapeOpt shape (const marsh::Maps& attrs, const ShapesT& shapes)
	{
		teq::RankT return_dim;
		eigen::Packer<teq::RankT>().unpack(return_dim, attrs);

		teq::ShapeSignature shape = shapes.front();
		if (shape.at(return_dim) == 1)
		{
			logs::debugf("argreducing with no significant dimensions... "
				"treating as identity: (return_dim=%d, shape=%s)",
				(int) return_dim, shape.to_string().c_str());
			return ShapeOpt();
		}

		std::vector<teq::DimT> slist = std::vector<teq::DimT>(shape.begin(), shape.end());
		slist[return_dim] = 1;
		return teq::ShapeSignature(slist);
	}
};

template <>
struct ShapeParser<egen::SLICE> final
{
	ShapeOpt shape (const marsh::Maps& attrs, const ShapesT& shapes)
	{
		eigen::PairVecT<teq::DimT> extents;
		eigen::Packer<eigen::PairVecT<teq::DimT>>().unpack(extents, attrs);

		teq::ShapeSignature shape = shapes.front();
		std::vector<teq::DimT> slist(shape.begin(), shape.end());
		for (size_t i = 0, n = extents.size(); i < n; ++i)
		{
			teq::DimT offsets = extents[i].first;
			if (offsets < shape.at(i))
			{
				slist[i] = std::min(extents[i].second, (teq::DimT) (shape.at(i) - offsets));
			}
		}

		teq::ShapeSignature sign(slist);
		if (std::all_of(slist.begin(), slist.end(),
			[](teq::DimT d) { return d > 0; }) &&
			sign.compatible_after(shape, 0))
		{
			logs::debugf("slice parameter covers whole tensor... "
				"treating as identity: (extents=%s)",
				eigen::to_string(extents).c_str());
			return ShapeOpt();
		}
		return sign;
	}
};

template <>
struct ShapeParser<egen::PAD> final
{
	ShapeOpt shape (const marsh::Maps& attrs, const ShapesT& shapes)
	{
		eigen::PairVecT<teq::DimT> paddings;
		eigen::Packer<eigen::PairVecT<teq::DimT>>().unpack(paddings, attrs);

		if (std::all_of(paddings.begin(), paddings.end(),
			[](std::pair<teq::DimT,teq::DimT> pad)
			{
				return pad.first == 0 && pad.second == 0;
			}))
		{
			logs::debugf("padding are all zero... "
				"treating as identity: (paddings=%s)",
				eigen::to_string(paddings).c_str());
			return ShapeOpt();
		}

		teq::ShapeSignature shape = shapes.front();
		std::vector<teq::DimT> slist(shape.begin(), shape.end());
		size_t n = std::min(paddings.size(), (size_t) teq::rank_cap);
		for (size_t i = 0; i < n; ++i)
		{
			if (slist[i] > 0)
			{
				slist[i] += paddings[i].first + paddings[i].second;
			}
		}
		return teq::ShapeSignature(slist);
	}
};

template <>
struct ShapeParser<egen::STRIDE> final
{
	ShapeOpt shape (const marsh::Maps& attrs, const ShapesT& shapes)
	{
		std::vector<teq::DimT> incrs;
		eigen::Packer<std::vector<teq::DimT>>().unpack(incrs, attrs);

		teq::ShapeSignature shape = shapes.front();
		std::vector<double> coords(teq::rank_cap, 1);
		size_t n = std::min(incrs.size(), (size_t) teq::rank_cap);
		for (size_t i = 0; i < n; ++i)
		{
			coords[i] = incrs[i];
		}
		std::vector<teq::DimT> slist(shape.begin(), shape.end());
		for (size_t i = 0; i < n; ++i)
		{
			slist[i] = std::round((double) slist[i] / incrs[i]);
		}
		return teq::ShapeSignature(slist);
	}
};

template <>
struct ShapeParser<egen::SCATTER> final
{
	ShapeOpt shape (const marsh::Maps& attrs, const ShapesT& shapes)
	{
		teq::Shape outshape;
		eigen::Packer<teq::Shape>().unpack(outshape, attrs);

		teq::ShapeSignature inshape = shapes.front();
		teq::ShapeSignature sign(
			std::vector<teq::DimT>(outshape.begin(), outshape.end()));
		if (std::all_of(inshape.begin(), inshape.end(),
			[](teq::DimT d) { return d > 0; }) &&
			inshape.compatible_after(sign, 0))
		{
			logs::debugf("scattering produces the same shape %s",
				outshape.to_string().c_str());
			return ShapeOpt();
		}
		return sign;
	}
};

template <>
struct ShapeParser<egen::MATMUL> final
{
	ShapeOpt shape (const marsh::Maps& attrs, const ShapesT& shapes)
	{
		eigen::PairVecT<teq::RankT> ranks;
		eigen::Packer<eigen::PairVecT<teq::RankT>>().unpack(ranks, attrs);

		// check common dimensions
		std::array<bool,teq::rank_cap> avisit;
		std::array<bool,teq::rank_cap> bvisit;
		std::fill(avisit.begin(), avisit.end(), false);
		std::fill(bvisit.begin(), bvisit.end(), false);
		teq::ShapeSignature ashape = shapes[0];
		teq::ShapeSignature bshape = shapes[1];
		for (const std::pair<teq::RankT,teq::RankT>& coms : ranks)
		{
			if (ashape.at(coms.first) != bshape.at(coms.second))
			{
				logs::fatalf("invalid shapes %s and %s do not match "
					"common dimensions %s", ashape.to_string().c_str(),
					bshape.to_string().c_str(),
					eigen::to_string(ranks).c_str());
			}
			if (avisit[coms.first] || bvisit[coms.second])
			{
				logs::fatalf("contraction dimensions %s must be unique for "
					"each side", eigen::to_string(ranks).c_str());
			}
			avisit[coms.first] = bvisit[coms.second] = true;
		}
		std::vector<teq::DimT> alist = teq::narrow_shape(ashape);
		std::vector<teq::DimT> blist = teq::narrow_shape(bshape);
		std::vector<teq::DimT> outlist;
		outlist.reserve(2 * ranks.size());
		for (teq::RankT i = 0, n = blist.size(); i < n; ++i)
		{
			if (false == bvisit[i])
			{
				outlist.push_back(blist.at(i));
			}
		}
		for (teq::RankT i = 0, n = alist.size(); i < n; ++i)
		{
			if (false == avisit[i])
			{
				outlist.push_back(alist.at(i));
			}
		}
		size_t nout = outlist.size();
		if ((teq::is_ambiguous(ashape) || teq::is_ambiguous(bshape)) &&
			nout < teq::rank_cap)
		{
			outlist.insert(outlist.end(), teq::rank_cap - nout, 0);
		}
		return teq::ShapeSignature(outlist);
	}
};

template <>
struct ShapeParser<egen::CONV> final
{
	ShapeOpt shape (const marsh::Maps& attrs, const ShapesT& shapes)
	{
		std::vector<teq::RankT> ranks;
		eigen::Packer<std::vector<teq::RankT>>().unpack(ranks, attrs);

		size_t n = std::min(ranks.size(), (size_t) teq::rank_cap);
		teq::ShapeSignature kernelshape = shapes[1];
		if (std::any_of(kernelshape.begin() + n, kernelshape.end(),
			[](teq::DimT d) { return d > 1; }))
		{
			logs::fatalf("cannot have ambiguous ranks not specified in "
				"kernelshape %s (ranks=%s)", kernelshape.to_string().c_str(),
				fmts::to_string(ranks.begin(), ranks.end()).c_str());
		}
		teq::ShapeSignature imgshape = shapes[0];
		std::vector<teq::DimT> slist(imgshape.begin(), imgshape.end());
		for (size_t i = 0; i < n; ++i)
		{
			teq::DimT& sdim = slist[ranks[i]];
			teq::DimT kdim = kernelshape.at(i);
			// treat as ambiguous if either dimension is ambiguous
			if (0 == sdim || 0 == kdim)
			{
				sdim = 0;
			}
			else
			{
				if (kdim > sdim)
				{
					logs::fatalf("cannot convolve a kernel of shape %s against "
						"smaller image of shape %s",
						kernelshape.to_string().c_str(),
						imgshape.to_string().c_str());
				}
				sdim -= kdim - 1;
			}
		}
		return teq::ShapeSignature(slist);
	}
};

template <>
struct ShapeParser<egen::REVERSE> final
{
	ShapeOpt shape (const marsh::Maps& attrs, const ShapesT& shapes)
	{
		return shapes.front();
	}
};

static inline bool is_inorder (const std::vector<teq::RankT>& order)
{
	size_t n = order.size();
	bool inorder = n > 0 ? (order[0] == 0) : true;
	for (size_t i = 1; i < n && inorder; ++i)
	{
		inorder = inorder && (order[i] == (order[i-1] + 1));
	}
	return inorder;
}

template <>
struct ShapeParser<egen::PERMUTE> final
{
	ShapeOpt shape (const marsh::Maps& attrs, const ShapesT& shapes)
	{
		std::vector<teq::RankT> order;
		eigen::Packer<std::vector<teq::RankT>>().unpack(order, attrs);

		if (is_inorder(order))
		{
			logs::debug("permuting with same "
				"dimensions ... treating as identity");
			return ShapeOpt();
		}
		bool visited[teq::rank_cap];
		std::fill(visited, visited + teq::rank_cap, false);
		for (teq::RankT i = 0, n = order.size(); i < n; ++i)
		{
			if (visited[order[i]])
			{
				logs::fatalf("permute does not support repeated orders "
					"(order=%s)", fmts::to_string(
						order.begin(), order.end()).c_str());
			}
			visited[order[i]] = true;
		}
		for (teq::RankT i = 0; i < teq::rank_cap; ++i)
		{
			if (false == visited[i])
			{
				order.push_back(i);
			}
		}
		teq::ShapeSignature shape = shapes.front();
		std::vector<teq::DimT> slist(teq::rank_cap, 0);
		for (teq::RankT i = 0; i < teq::rank_cap; ++i)
		{
			slist[i] = shape.at(order[i]);
		}
		return teq::ShapeSignature(slist);
	}
};

template <>
struct ShapeParser<egen::EXTEND> final
{
	ShapeOpt shape (const marsh::Maps& attrs, const ShapesT& shapes)
	{
		std::vector<teq::DimT> bcast;
		eigen::Packer<std::vector<teq::DimT>>().unpack(bcast, attrs);

		if (bcast.empty() || std::all_of(bcast.begin(), bcast.end(),
			[](teq::DimT d) { return 1 == d; }))
		{
			logs::debug("extending with nothing... treating as identity");
			return ShapeOpt();
		}
		if (std::any_of(bcast.begin(), bcast.end(),
			[](teq::DimT d) { return 0 == d; }))
		{
			logs::fatalf("cannot extend using zero dimensions %s",
				fmts::to_string(bcast.begin(), bcast.end()).c_str());
		}
		teq::ShapeSignature shape = shapes.front();
		std::vector<teq::DimT> slist(shape.begin(), shape.end());
		for (size_t i = 0, nbcasts = bcast.size(); i < nbcasts; ++i)
		{
			if (bcast.at(i) > 1 && shape.at(i) > 1)
			{
				logs::fatalf("cannot extend non-singular dimension %d of "
					"shape %s: bcast=%s", i, shape.to_string().c_str(),
					fmts::to_string(bcast.begin(), bcast.end()).c_str());
			}
			slist[i] *= bcast[i];
		}
		return teq::ShapeSignature(slist);
	}
};

template <>
struct ShapeParser<egen::CONCAT> final
{
	ShapeOpt shape (const marsh::Maps& attrs, const ShapesT& shapes)
	{
		teq::RankT axis;
		eigen::Packer<teq::RankT>().unpack(axis, attrs);

		teq::ShapeSignature leftshape = shapes[0];
		teq::ShapeSignature rightshape = shapes[1];
		std::vector<teq::DimT> slist(leftshape.begin(), leftshape.end());
		if (slist[axis] == 0 || rightshape.at(axis) == 0)
		{
			slist[axis] = 0;
		}
		else
		{
			slist[axis] += rightshape.at(axis);
		}
		return teq::ShapeSignature(slist);
	}
};

template <>
struct ShapeParser<egen::GROUP_CONCAT> final
{
	ShapeOpt shape (const marsh::Maps& attrs, const ShapesT& shapes)
	{
		if (shapes.size() == 1)
		{
			logs::debug("concatenating a single node... treating as identity");
			return ShapeOpt();
		}
		teq::RankT axis;
		eigen::Packer<teq::RankT>().unpack(axis, attrs);

		if (std::any_of(shapes.begin(), shapes.end(),
			[axis](teq::ShapeSignature shape)
			{ return shape.at(axis) != 1; }))
		{
			logs::fatal("cannot group concat shapes "
				"with dimension that is not one");
		}
		teq::ShapeSignature initshape = shapes.front();
		std::vector<teq::DimT> slist(initshape.begin(), initshape.end());
		slist[axis] = shapes.size();
		return teq::ShapeSignature(slist);
	}
};

template <>
struct ShapeParser<egen::RESHAPE> final
{
	ShapeOpt shape (const marsh::Maps& attrs, const ShapesT& shapes)
	{
		teq::Shape outshape;
		eigen::Packer<teq::Shape>().unpack(outshape, attrs);

		teq::ShapeSignature inshape = shapes.front();
		teq::ShapeSignature sign(
			std::vector<teq::DimT>(outshape.begin(), outshape.end()));
		if (std::all_of(inshape.begin(), inshape.end(),
			[](teq::DimT d) { return d > 0; }) &&
			inshape.compatible_after(sign, 0))
		{
			logs::debugf("reshape produces the same shape %s",
				outshape.to_string().c_str());
			return ShapeOpt();
		}
		return sign;
	}
};

}

#endif // ETEQ_SHAPER_HPP
