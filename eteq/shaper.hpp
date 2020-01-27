#include "eteq/funcopt.hpp"

#ifndef ETEQ_SHAPER_HPP
#define ETEQ_SHAPER_HPP

namespace eteq
{

// todo: move these to eigen and auto-generate
template <egen::_GENERATED_OPCODE OPCODE>
struct ShapeParser final
{
	/// Return output shape if operator is not redundant
	/// given specified attributes and input shapes
	teq::Shape shape (const marsh::Maps& attrs, const teq::ShapesT& shapes)
	{
		if (shapes.empty())
		{
			teq::fatal("cannot generated outshape without shapes");
		}
		teq::Shape outshape = shapes.front();
		for (size_t i = 1, n = shapes.size(); i < n; ++i)
		{
			if (false == shapes[i].compatible_after(outshape, 0))
			{
				teq::fatalf("cannot %s with incompatible shapes %s and %s",
					egen::name_op(OPCODE).c_str(),
					shapes[i].to_string().c_str(),
					outshape.to_string().c_str());
			}
		}
		return outshape;
	}
};

struct ReducePacker
{
	virtual ~ReducePacker (void) = default;

	teq::Shape shape (const marsh::Maps& attrs, const teq::ShapesT& shapes)
	{
		std::set<teq::RankT> ranks;
		eigen::Packer<std::set<teq::RankT>>().unpack(ranks, attrs);
		teq::Shape shape = shapes.front();
		std::vector<teq::DimT> slist(shape.begin(), shape.end());
		for (teq::RankT i : ranks)
		{
			slist[i] = 1;
		}
		return teq::Shape(slist);
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
	teq::Shape shape (const marsh::Maps& attrs, const teq::ShapesT& shapes)
	{
		teq::RankT return_dim;
		eigen::Packer<teq::RankT>().unpack(return_dim, attrs);
		teq::Shape shape = shapes.front();
		std::vector<teq::DimT> slist(shape.begin(), shape.end());
		slist[return_dim] = 1;
		return teq::Shape(slist);
	}
};

template <>
struct ShapeParser<egen::SLICE> final
{
	teq::Shape shape (const marsh::Maps& attrs, const teq::ShapesT& shapes)
	{
		eigen::PairVecT<teq::DimT> extents;
		eigen::Packer<eigen::PairVecT<teq::DimT>>().unpack(extents, attrs);
		teq::Shape shape = shapes.front();
		std::vector<teq::DimT> slist(shape.begin(), shape.end());
		for (size_t i = 0, n = std::min(extents.size(),
			(size_t) teq::rank_cap); i < n; ++i)
		{
			teq::DimT offsets = extents[i].first;
			if (offsets < shape.at(i))
			{
				slist[i] = std::min(extents[i].second, (teq::DimT) (shape.at(i) - offsets));
			}
		}
		return teq::Shape(slist);
	}
};

template <>
struct ShapeParser<egen::PAD> final
{
	teq::Shape shape (const marsh::Maps& attrs, const teq::ShapesT& shapes)
	{
		eigen::PairVecT<teq::DimT> paddings;
		eigen::Packer<eigen::PairVecT<teq::DimT>>().unpack(paddings, attrs);
		teq::Shape shape = shapes.front();
		std::vector<teq::DimT> slist(shape.begin(), shape.end());
		for (size_t i = 0, n = std::min(paddings.size(),
			(size_t) teq::rank_cap); i < n; ++i)
		{
			if (slist[i] > 0)
			{
				slist[i] += paddings[i].first + paddings[i].second;
			}
		}
		return teq::Shape(slist);
	}
};

template <>
struct ShapeParser<egen::STRIDE> final
{
	teq::Shape shape (const marsh::Maps& attrs, const teq::ShapesT& shapes)
	{
		std::vector<teq::DimT> incrs;
		eigen::Packer<std::vector<teq::DimT>>().unpack(incrs, attrs);

		teq::Shape shape = shapes.front();
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
		return teq::Shape(slist);
	}
};

template <>
struct ShapeParser<egen::SCATTER> final
{
	teq::Shape shape (const marsh::Maps& attrs, const teq::ShapesT& shapes)
	{
		teq::Shape outshape;
		eigen::Packer<teq::Shape>().unpack(outshape, attrs);
		return outshape;
	}
};

template <>
struct ShapeParser<egen::MATMUL> final
{
	teq::Shape shape (const marsh::Maps& attrs, const teq::ShapesT& shapes)
	{
		eigen::PairVecT<teq::RankT> ranks;
		eigen::Packer<eigen::PairVecT<teq::RankT>>().unpack(ranks, attrs);

		// check common dimensions
		std::array<bool,teq::rank_cap> avisit;
		std::array<bool,teq::rank_cap> bvisit;
		std::fill(avisit.begin(), avisit.end(), false);
		std::fill(bvisit.begin(), bvisit.end(), false);
		teq::Shape ashape = shapes[0];
		teq::Shape bshape = shapes[1];
		for (const std::pair<teq::RankT,teq::RankT>& coms : ranks)
		{
			if (ashape.at(coms.first) != bshape.at(coms.second))
			{
				teq::fatalf("invalid shapes %s and %s do not match "
					"common dimensions %s", ashape.to_string().c_str(),
					bshape.to_string().c_str(),
					eigen::to_string(ranks).c_str());
			}
			if (avisit[coms.first] || bvisit[coms.second])
			{
				teq::fatalf("contraction dimensions %s must be unique for "
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
		return teq::Shape(outlist);
	}
};

template <>
struct ShapeParser<egen::CONV> final
{
	teq::Shape shape (const marsh::Maps& attrs, const teq::ShapesT& shapes)
	{
		std::vector<teq::RankT> ranks;
		eigen::Packer<std::vector<teq::RankT>>().unpack(ranks, attrs);

		size_t n = std::min(ranks.size(), (size_t) teq::rank_cap);
		teq::Shape kernelshape = shapes[1];
		if (std::any_of(kernelshape.begin() + n, kernelshape.end(),
			[](teq::DimT d) { return d > 1; }))
		{
			teq::fatalf("cannot have ambiguous ranks not specified in "
				"kernelshape %s (ranks=%s)", kernelshape.to_string().c_str(),
				fmts::to_string(ranks.begin(), ranks.end()).c_str());
		}
		teq::Shape imgshape = shapes[0];
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
					teq::fatalf("cannot convolve a kernel of shape %s against "
						"smaller image of shape %s",
						kernelshape.to_string().c_str(),
						imgshape.to_string().c_str());
				}
				sdim -= kdim - 1;
			}
		}
		return teq::Shape(slist);
	}
};

template <>
struct ShapeParser<egen::REVERSE> final
{
	teq::Shape shape (const marsh::Maps& attrs, const teq::ShapesT& shapes)
	{
		return shapes.front();
	}
};

template <>
struct ShapeParser<egen::PERMUTE> final
{
	teq::Shape shape (const marsh::Maps& attrs, const teq::ShapesT& shapes)
	{
		std::vector<teq::RankT> order;
		eigen::Packer<std::vector<teq::RankT>>().unpack(order, attrs);
		bool visited[teq::rank_cap];
		std::fill(visited, visited + teq::rank_cap, false);
		for (teq::RankT i = 0, n = std::min(order.size(),
			(size_t) teq::rank_cap); i < n; ++i)
		{
			if (visited[order[i]])
			{
				teq::fatalf("permute does not support repeated orders "
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
		teq::Shape shape = shapes.front();
		std::vector<teq::DimT> slist(teq::rank_cap, 1);
		for (teq::RankT i = 0; i < teq::rank_cap; ++i)
		{
			slist[i] = shape.at(order[i]);
		}
		return teq::Shape(slist);
	}
};

template <>
struct ShapeParser<egen::EXTEND> final
{
	teq::Shape shape (const marsh::Maps& attrs, const teq::ShapesT& shapes)
	{
		teq::Shape shape = shapes.front();
		std::vector<teq::DimT> bcast = eigen::unpack_extend(shape, attrs);
		if (std::any_of(bcast.begin(), bcast.end(),
			[](teq::DimT d) { return 0 == d; }))
		{
			teq::fatalf("cannot extend using zero dimensions %s",
				fmts::to_string(bcast.begin(), bcast.end()).c_str());
		}
		std::vector<teq::DimT> slist(shape.begin(), shape.end());
		for (size_t i = 0, nbcasts = bcast.size(); i < nbcasts; ++i)
		{
			if (bcast.at(i) > 1 && shape.at(i) > 1)
			{
				teq::fatalf("cannot extend non-singular dimension %d of "
					"shape %s: bcast=%s", i, shape.to_string().c_str(),
					fmts::to_string(bcast.begin(), bcast.end()).c_str());
			}
			slist[i] *= bcast[i];
		}
		return teq::Shape(slist);
	}
};

template <>
struct ShapeParser<egen::CONCAT> final
{
	teq::Shape shape (const marsh::Maps& attrs, const teq::ShapesT& shapes)
	{
		teq::RankT axis;
		eigen::Packer<teq::RankT>().unpack(axis, attrs);
		if (shapes.size() > 2)
		{
			if (std::any_of(shapes.begin(), shapes.end(),
				[axis](teq::Shape shape)
				{ return shape.at(axis) != 1; }))
			{
				teq::fatal("cannot group concat shapes "
					"with dimension that is not one");
			}
			teq::Shape initshape = shapes.front();
			std::vector<teq::DimT> slist(initshape.begin(), initshape.end());
			slist[axis] = shapes.size();
			return teq::Shape(slist);
		}
		teq::Shape leftshape = shapes[0];
		teq::Shape rightshape = shapes[1];
		std::vector<teq::DimT> slist(leftshape.begin(), leftshape.end());
		if (slist[axis] == 0 || rightshape.at(axis) == 0)
		{
			slist[axis] = 0;
		}
		else
		{
			slist[axis] += rightshape.at(axis);
		}
		return teq::Shape(slist);
	}
};

template <>
struct ShapeParser<egen::RESHAPE> final
{
	teq::Shape shape (const marsh::Maps& attrs, const teq::ShapesT& shapes)
	{
		teq::Shape outshape;
		eigen::Packer<teq::Shape>().unpack(outshape, attrs);
		return outshape;
	}
};

}

#endif // ETEQ_SHAPER_HPP
