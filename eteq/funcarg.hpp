//
/// funcarg.hpp
/// eteq
///
/// Purpose:
/// Typed Eigen implementation of teq iEdge
///

#include "eigen/operator.hpp"

#include "eteq/inode.hpp"

#ifndef ETEQ_FUNCARG_HPP
#define ETEQ_FUNCARG_HPP

namespace eteq
{

/// Implementation of iEigenEdge using node as tensor wrapper
template <typename T>
struct FuncArg final : public eigen::iEigenEdge<T>
{
	FuncArg (NodeptrT<T> node) :
		node_(node)
	{
		if (node_ == nullptr)
		{
			logs::fatal("cannot map a null node");
		}
		shape_ = node->shape();
	}

	FuncArg (NodeptrT<T> node, teq::Shape shape,
		std::vector<double> coords) :
		node_(node), shape_(shape), coords_(coords)
	{
		if (node_ == nullptr)
		{
			logs::fatal("cannot map a null node");
		}
	}

	/// Implementation of iEdge
	teq::Shape shape (void) const override
	{
		return shape_;
	}

	/// Implementation of iEdge
	teq::Shape argshape (void) const override
	{
		return node_->shape();
	}

	/// Implementation of iEdge
	teq::TensptrT get_tensor (void) const override
	{
		return node_->get_tensor();
	}

	/// Implementation of iEdge
	void get_attrs (marsh::Maps& out) const override
	{
		if (false == shape_.compatible_after(node_->shape(), 0))
		{
			auto arr = std::make_unique<marsh::NumArray<double>>();
			arr->contents_ = std::vector<double>(shape_.begin(), shape_.end());
			out.contents_.emplace(eigen::shaper_key, std::move(arr));
		}
		if (coords_.size() > 0)
		{
			auto arr = std::make_unique<marsh::NumArray<double>>();
			arr->contents_ = coords_;
			out.contents_.emplace(eigen::coorder_key, std::move(arr));
		}
	}

	/// Implementation of iEigenEdge<T>
	T* data (void) const override
	{
		return node_->data();
	}

	void set_tensor (teq::TensptrT tens)
	{
		node_ = to_node<T>(tens);
	}

	NodeptrT<T> get_node (void) const
	{
		return node_;
	}

private:
	/// Tensor reference
	NodeptrT<T> node_;

	/// Output shape
	teq::Shape shape_;

	/// Coordinate transformation attributes
	std::vector<double> coords_;
};

/// Type of typed functor arguments
template <typename T>
using ArgsT = std::vector<FuncArg<T>>;

/// Return FuncArg<T> that reduces input tensor by
/// units in reduction vector after specified rank
/// E.g.: tensor w/ shape [2, 3, 4], offset = 1, ndims = 2
/// gets mapped to [2, 1, 1]
template <typename T>
FuncArg<T> reduce_map (NodeptrT<T> node, teq::RankT offset, teq::RankT ndims)
{
	if (offset >= teq::rank_cap)
	{
		logs::fatalf("cannot reduce dimensions [%d:]. Must be less than %d",
			offset, teq::rank_cap);
	}

	teq::Shape shape = node->shape();
	std::vector<teq::DimT> slist(shape.begin(), shape.end());
	std::set<teq::RankT> dims; // dims are allowed to be non-contiguous
	for (size_t i = offset,
		n = std::min((size_t) offset + ndims, (size_t) teq::rank_cap);
		i < n; ++i)
	{
		if (shape.at(i) > 1)
		{
			dims.emplace(i);
			slist[i] = 1;
		}
	}

	std::vector<double> rdims(teq::rank_cap, teq::rank_cap);
	std::copy(dims.begin(), dims.end(), rdims.begin());
	return FuncArg<T>(node, teq::Shape(slist), rdims);
}

/// Return FuncArg<T> that reduce tensor by argument index
/// return_dim greater than rank_cap looks across all dimensions
template <typename T>
FuncArg<T> argreduce_map (NodeptrT<T> node, teq::RankT return_dim)
{
	std::vector<teq::DimT> slist;
	if (return_dim < teq::rank_cap)
	{
		teq::Shape shape = node->shape();
		std::copy(shape.begin(), shape.end(), std::back_inserter(slist));
		slist[return_dim] = 1;
	}

	return FuncArg<T>(node, teq::Shape(slist), std::vector<double>{
		static_cast<double>(return_dim)});
}

/// Return FuncArg<T> that extends input tensor by
/// rank and extension vector
/// E.g.: tensor w/ shape [2, 1, 1], rank = 1, ext = [3, 4]
/// gets mapped to [2, 3, 4]
template <typename T>
FuncArg<T> extend_map (NodeptrT<T> node,
	teq::RankT rank, std::vector<teq::DimT> ext)
{
	size_t n_ext = ext.size();
	if (0 == n_ext)
	{
		logs::warn("extending with empty vector ... will do nothing");
		return FuncArg<T>(node);
	}
	if (std::any_of(ext.begin(), ext.end(),
		[](teq::DimT& d) { return 0 == d; }))
	{
		logs::fatalf("cannot extend using zero dimensions %s",
			fmts::to_string(ext.begin(), ext.end()).c_str());
	}
	if (rank + n_ext > teq::rank_cap)
	{
		logs::fatalf("cannot extend shape rank %d beyond rank_cap with n_ext %d",
			rank, n_ext);
	}
	std::vector<double> bcast(teq::rank_cap, 1);
	auto it = bcast.begin();
	std::copy(ext.begin(), ext.end(), it + rank);

	teq::Shape shape = node->shape();
	std::vector<teq::DimT> slist(shape.begin(), shape.end());
	for (teq::DimT e : ext)
	{
		slist[rank] *= e;
		++rank;
	}

	return FuncArg<T>(node, teq::Shape(slist), bcast);
}

/// Return FuncArg<T> that permutes input tensor by order
/// E.g.: tensor w/ shape [2, 3, 4], order = [1, 2, 0]
/// gets mapped to [3, 4, 2]
template <typename T>
FuncArg<T> permute_map (NodeptrT<T> node, std::vector<teq::RankT> order)
{
	if (order.size() == 0)
	{
		logs::warn("permuting with same dimensions ... will do nothing");
		return FuncArg<T>(node);
	}

	bool visited[teq::rank_cap];
	std::fill(visited, visited + teq::rank_cap, false);
	for (teq::RankT i = 0, n = order.size(); i < n; ++i)
	{
		if (visited[order[i]])
		{
			logs::fatalf("permute does not support repeated orders: %s",
				fmts::to_string(order.begin(), order.end()).c_str());
		}
		visited[order[i]] = true;
	}
	// since order can't be duplicate, norder < rank_cap
	for (teq::RankT i = 0; i < teq::rank_cap; ++i)
	{
		if (false == visited[i])
		{
			order.push_back(i);
		}
	}

	teq::Shape shape = node->shape();
	std::vector<teq::DimT> slist(teq::rank_cap, 0);
	for (teq::RankT i = 0; i < teq::rank_cap; ++i)
	{
		slist[i] = shape.at(order[i]);
	}

	return FuncArg<T>(node, teq::Shape(slist), std::vector<double>(order.begin(), order.end()));
}

/// Return FuncArg<T> that reshapes node to specified shape
template <typename T>
FuncArg<T> reshape_map (NodeptrT<T> node, const teq::Shape& shape)
{
	return FuncArg<T>(node, shape, {});
}

/// Return FuncArg<T> that takes specific slice of tensor according to
/// vector of offset, extent pairs
/// E.g.: tensor w/ shape [2, 3, 4], offset = 1, extent = 2, and dimension = 2
/// gets mapped to [2, 3, 2] that references [:,:,1:3]
/// (second and third slices of the 3rd dimension)
template <typename T>
FuncArg<T> slice_map (NodeptrT<T> node, const eigen::PairVecT<teq::DimT>& extents)
{
	if (extents.size() > teq::rank_cap)
	{
		eigen::PairVecT<int> readable_extents(extents.begin(), extents.end());
		logs::fatalf(
			"cannot slice dimensions beyond rank_cap %d: using extent %s",
			teq::rank_cap,
			fmts::to_string(readable_extents.begin(), readable_extents.end()).c_str());
	}
	teq::Shape shape = node->shape();
	std::vector<teq::DimT> slist(shape.begin(), shape.end());
	slist.reserve(teq::rank_cap);
	for (size_t i = 0,  n = extents.size(); i < n; ++i)
	{
		auto& ex = extents[i];
		if (ex.second < 1)
		{
			eigen::PairVecT<int> readable_extents(extents.begin(), extents.end());
			logs::fatalf("cannot extend zero slices: extents %s",
				fmts::to_string(readable_extents.begin(), readable_extents.end()).c_str());
		}
		teq::DimT offset = std::min(ex.first, (teq::DimT) (shape.at(i) - 1));
		slist[i] = std::min(ex.second, (teq::DimT) (shape.at(i) - offset));
	}
	return FuncArg<T>(node, teq::Shape(slist), eigen::encode_pair(extents));
}

/// Return FuncArg<T> that pads tensor with 0s across specified dimensions
/// E.g.: tensor w/ shape [2, 3, 4], padding = {2,1}, dimension = 0
/// gets mapped to [5, 3, 4] where [0,:,:] and [3:5,:,:] are 0
/// (first, fourth, and fifth slices of the 1st dimension are 0)
template <typename T>
FuncArg<T> pad_map (NodeptrT<T> node, const eigen::PairVecT<teq::DimT>& paddings)
{
	if (paddings.size() > teq::rank_cap)
	{
		eigen::PairVecT<int> readable_paddings(paddings.begin(), paddings.end());
		logs::fatalf(
			"cannot pad dimensions beyond rank_cap %d: using paddings %s",
			teq::rank_cap,
			fmts::to_string(readable_paddings.begin(), readable_paddings.end()).c_str());
	}
	teq::Shape shape = node->shape();
	std::vector<teq::DimT> slist(shape.begin(), shape.end());
	size_t n = std::min(paddings.size(), (size_t) teq::rank_cap);
	for (size_t i = 0; i < n; ++i)
	{
		slist[i] += paddings[i].first + paddings[i].second;
	}
	return FuncArg<T>(node, teq::Shape(slist), eigen::encode_pair(paddings));
}

/// Return FuncArg<T> that takes elements of
/// specific increments across dimensions starting from 0
/// E.g.: tensor w/ shape [2, 3, 4], incrs = {1, 2, 2}
/// gets mapped to [2, 2, 2] where
/// output[:,0,0] takes on input[:,0,0]
/// output[:,1,0] takes on input[:,2,0]
/// output[:,0,1] takes on input[:,0,2]
/// output[:,1,1] takes on input[:,2,2]
template <typename T>
FuncArg<T> stride_map (NodeptrT<T> node,
	const std::vector<teq::DimT>& incrs)
{
	if (incrs.size() > teq::rank_cap)
	{
		logs::warnf("trying to stride in dimensions beyond rank_cap %d: "
			"using increments %s (will ignore those dimensions)", teq::rank_cap,
			fmts::to_string(incrs.begin(), incrs.end()).c_str());
	}
	std::vector<double> coords(teq::rank_cap, 1);
	size_t n = std::min(incrs.size(), (size_t) teq::rank_cap);
	for (size_t i = 0; i < n; ++i)
	{
		coords[i] = incrs[i];
	}
	teq::Shape shape = node->shape();
	std::vector<teq::DimT> slist(shape.begin(), shape.end());
	for (size_t i = 0; i < n; ++i)
	{
		slist[i] = std::round((double) slist[i] / incrs[i]);
	}
	return FuncArg<T>(node, teq::Shape(slist), coords);
}

template <typename T>
ArgsT<T> convolve_map (NodeptrT<T> image, NodeptrT<T> kernel,
	const std::vector<teq::RankT>& dims)
{
	teq::Shape inshape = image->shape();
	teq::Shape kernelshape = kernel->shape();

	size_t n = std::min(dims.size(), (size_t) teq::rank_cap);
	if (std::any_of(kernelshape.begin() + n, kernelshape.end(),
		[](teq::DimT d)
		{
			return d > 1;
		}))
	{
		logs::fatalf("invalid kernelshape %s does not solely match dimensions %s",
			kernelshape.to_string().c_str(),
			fmts::to_string(dims.begin(), dims.end()).c_str());
	}

	std::vector<teq::DimT> slist(inshape.begin(), inshape.end());
	for (size_t i = 0; i < n; ++i)
	{
		slist[dims[i]] -= kernelshape.at(i) - 1;
	}
	teq::Shape outshape(slist);

	std::vector<double> kernel_dims(teq::rank_cap, teq::rank_cap);
	auto it = kernel_dims.begin();
	std::copy(dims.begin(), dims.end(), it);
	return {
		FuncArg<T>(image, outshape, {}),
		FuncArg<T>(kernel, outshape, kernel_dims),
	};
}

template <typename T>
ArgsT<T> concat_map (NodeptrT<T> left, NodeptrT<T> right, teq::RankT axis)
{
	teq::Shape leftshape = left->shape();
	teq::Shape rightshape = right->shape();
	std::vector<teq::DimT> slist(leftshape.begin(), leftshape.end());
	slist[axis] += rightshape.at(axis);
	teq::Shape outshape(slist);
	return {
		FuncArg<T>(left, outshape, std::vector<double>{
			static_cast<double>(axis)}),
		FuncArg<T>(right, outshape, {}),
	};
}

template <typename T>
ArgsT<T> group_concat_map (NodesT<T> args, teq::RankT axis)
{
	size_t nargs = args.size();
	if (nargs < 2)
	{
		logs::fatal("cannot group concat less than 2 arguments");
	}
	if (std::any_of(args.begin(), args.end(),
		[](NodeptrT<T> arg) { return nullptr == arg; }))
	{
		logs::fatal("cannot group concat with null argument");
	}
	teq::Shape initshape = args[0]->shape();
	std::vector<teq::DimT> slist(initshape.begin(), initshape.end());
	for (size_t i = 1; i < nargs; ++i)
	{
		slist[axis] += args[i]->shape().at(axis);
	}
	teq::Shape outshape(slist);

	ArgsT<T> out;
	out.reserve(nargs);
	out.push_back(FuncArg<T>(args[0], outshape, std::vector<double>{
		static_cast<double>(axis)}));
	std::transform(args.begin() + 1, args.end(), std::back_inserter(out),
		[&](NodeptrT<T> arg)
		{
			return FuncArg<T>(arg, outshape, {});
		});
	return out;
}

template <typename T>
ArgsT<T> contract_map (NodeptrT<T> a, NodeptrT<T> b, eigen::PairVecT<teq::RankT> dims)
{
	teq::Shape ashape = a->shape();
	teq::Shape bshape = b->shape();
	// check common dimensions
	std::array<bool,teq::rank_cap> avisit;
	std::array<bool,teq::rank_cap> bvisit;
	std::fill(avisit.begin(), avisit.end(), false);
	std::fill(bvisit.begin(), bvisit.end(), false);
	for (std::pair<teq::RankT,teq::RankT>& coms : dims)
	{
		if (ashape.at(coms.first) != bshape.at(coms.second))
		{
			eigen::PairVecT<int> readable_dims(dims.begin(), dims.end());
			logs::fatalf("invalid shapes %s and %s do not match common dimensions %s",
				ashape.to_string().c_str(), bshape.to_string().c_str(),
				fmts::to_string(readable_dims.begin(), readable_dims.end()).c_str());
		}
		if (avisit[coms.first] || bvisit[coms.second])
		{
			eigen::PairVecT<int> readable_dims(dims.begin(), dims.end());
			logs::fatalf("contraction dimensions %s must be unique for each side",
				fmts::to_string(readable_dims.begin(), readable_dims.end()).c_str());
		}
		avisit[coms.first] = bvisit[coms.second] = true;
	}
	std::vector<teq::DimT> alist = teq::narrow_shape(ashape);
	std::vector<teq::DimT> blist = teq::narrow_shape(bshape);
	std::vector<teq::DimT> outlist;
	outlist.reserve(teq::rank_cap);
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
	if (teq::rank_cap > outlist.size())
	{
		outlist.insert(outlist.end(), teq::rank_cap - outlist.size(), 1);
	}
	teq::Shape outshape(outlist);
	return {
		eteq::FuncArg<T>(a, outshape, eigen::encode_pair(dims)),
		eteq::FuncArg<T>(b, outshape, {}),
	};
}

}

#endif // ETEQ_FUNCARG_HPP
