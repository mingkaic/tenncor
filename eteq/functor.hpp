//
/// functor.hpp
/// eteq
///
/// Purpose:
/// Eigen functor implementation of operable func
///

#include "teq/iopfunc.hpp"

#include "tag/locator.hpp"

#include "eigen/generated/opcode.hpp"

#include "eteq/edge.hpp"

#ifndef ETEQ_FUNCTOR_HPP
#define ETEQ_FUNCTOR_HPP

namespace eteq
{

/// Functor implementation of operable functor of Eigen operators
template <typename T>
struct Functor final : public teq::iOperableFunc
{
	/// Return Functor given opcodes mapped to Eigen operators in operator.hpp
	static Functor<T>* get (egen::_GENERATED_OPCODE opcode, ArgsT<T> args);

	/// Return Functor move of other
	static Functor<T>* get (Functor<T>&& other)
	{
		return new Functor<T>(std::move(other));
	}

	Functor (const Functor<T>& other) = delete;

	Functor<T>& operator = (const Functor<T>& other) = delete;

	Functor<T>& operator = (Functor<T>&& other) = delete;

	/// Implementation of iTensor
	const teq::Shape& shape (void) const override
	{
		return shape_;
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return opcode_.name_;
	}

	/// Implementation of iFunctor
	teq::Opcode get_opcode (void) const override
	{
		return opcode_;
	}

	/// Implementation of iFunctor
	teq::CEdgesT get_children (void) const override
	{
		return teq::CEdgesT(args_.begin(), args_.end());
	}

	/// Implementation of iFunctor
	void update_child (teq::TensptrT arg, size_t index) override
	{
		teq::Shape nexshape = arg->shape();
		teq::Shape curshape = args_[index].argshape();
		if (false == nexshape.compatible_after(curshape, 0))
		{
			logs::fatalf("cannot update child %d to argument with "
				"incompatible shape %s (requires shape %s)",
				index, nexshape.to_string().c_str(),
				curshape.to_string().c_str());
		}
		static_cast<Edge<T>*>(&args_[index])->set_tensor(arg);
		// todo: warn of data destruction
		uninitialize();
	}

	/// Implementation of iOperableFunc
	void update (void) override
	{
		if (is_uninit())
		{
			initialize();
		}
		out_->assign();
	}

	/// Implementation of iData
	void* data (void) override
	{
		if (is_uninit())
		{
			initialize();
		}
		return out_->get_ptr();
	}

	/// Implementation of iData
	const void* data (void) const override
	{
		if (is_uninit())
		{
			logs::fatal("cannot get data of uninitialized functor");
		}
		return out_->get_ptr();
	}

	/// Implementation of iData
	size_t type_code (void) const override
	{
		return egen::get_type<T>();
	}

	/// Implementation of iData
	std::string type_label (void) const override
	{
		return egen::name_type(egen::get_type<T>());
	}

	/// Implementation of iData
	size_t nbytes (void) const override
	{
		return sizeof(T) * shape_.n_elems();
	}

	/// Return true if functor has never been initialized or
	/// was uninitialized, otherwise functor can return data
	bool is_uninit (void) const
	{
		return nullptr == out_;
	}

	/// Removes internal Eigen data object
	void uninitialize (void)
	{
		out_ = nullptr;
	}

	/// Populate internal Eigen data object
	void initialize (void)
	{
		egen::typed_exec<T>((egen::_GENERATED_OPCODE) opcode_.code_,
			out_, eigen::EigenEdgesT<T>(args_.begin(), args_.end()));
	}

private:
	Functor (egen::_GENERATED_OPCODE opcode, teq::Shape shape, ArgsT<T> args) :
		opcode_(teq::Opcode{egen::name_op(opcode), opcode}), shape_(shape), args_(args)
	{
#ifdef FINIT_ON_BUILD
		initialize();
#endif // FINIT_ON_BUILD
	}

	Functor (Functor<T>&& other) = default;

	eigen::EigenptrT<T> out_ = nullptr;

	/// Operation encoding
	teq::Opcode opcode_;

	/// Shape info built at construction time according to arguments
	teq::Shape shape_;

	/// Tensor arguments (and children)
	ArgsT<T> args_;
};

// todo: move these to eigen and auto-generate
/// Functor's node wrapper
template <typename T>
struct FunctorNode final : public iNode<T>
{
	FunctorNode (std::shared_ptr<Functor<T>> f) : func_(f) {}

	/// Return deep copy of this instance (with a copied functor)
	FunctorNode<T>* clone (void) const
	{
		return static_cast<FunctorNode<T>*>(clone_impl());
	}

	/// Implementation of iNode<T>
	T* data (void) override
	{
		return (T*) func_->data();
	}

	/// Implementation of iNode<T>
	void update (void) override
	{
		func_->update();
	}

	/// Implementation of iNode<T>
	teq::TensptrT get_tensor (void) const override
	{
		return func_;
	}

protected:
	iNode<T>* clone_impl (void) const override
	{
		auto args = func_->get_children();
		ArgsT<T> input_args;
		input_args.reserve(args.size());
		std::transform(args.begin(), args.end(),
			std::back_inserter(input_args),
			[](const teq::iEdge& arg) -> Edge<T>
			{
				return *static_cast<const Edge<T>*>(&arg);
			});
		return new FunctorNode(std::shared_ptr<Functor<T>>(Functor<T>::get(
			(egen::_GENERATED_OPCODE) func_->get_opcode().code_, input_args)));
	}

private:
	std::shared_ptr<Functor<T>> func_;
};

template <typename T>
Functor<T>* Functor<T>::get (egen::_GENERATED_OPCODE opcode, ArgsT<T> args)
{
	static bool registered = register_builder<Functor<T>,T>(
		[](teq::TensptrT tens)
		{
			return std::make_shared<FunctorNode<T>>(
				std::static_pointer_cast<Functor<T>>(tens));
		});
	assert(registered);

	size_t nargs = args.size();
	if (0 == nargs)
	{
		logs::fatalf("cannot perform `%s` with no arguments",
			egen::name_op(opcode).c_str());
	}

	teq::Shape shape = args[0].shape();
	for (size_t i = 1, n = nargs; i < n; ++i)
	{
		teq::Shape ishape = args[i].shape();
		if (false == ishape.compatible_after(shape, 0))
		{
			std::stringstream location;
			for (size_t i = 0; i < nargs; ++i)
			{
				location << "[" << i << "]\n"
					<< tag::display_location(args[i].get_tensor()) << "\n";
			}
			logs::fatalf("cannot perform `%s` with incompatible shapes %s "
				"and %s: childrens\n%s", egen::name_op(opcode).c_str(),
				shape.to_string().c_str(), ishape.to_string().c_str(),
				location.str().c_str());
		}
	}

	return new Functor<T>(opcode, shape, args);
}

template <typename T,egen::_GENERATED_OPCODE OPCODE>
struct FuncPacker final
{
	ArgsT<T> pack (const NodesT<T>& nodes)
	{
		ArgsT<T> args;
		args.reserve(nodes.size());
		std::transform(nodes.begin(), nodes.end(), std::back_inserter(args),
			[](NodeptrT<T> node) { return Edge<T>(node); });
		return args;
	}

	template <typename ...ARGS>
	ArgsT<T> pack (const NodesT<T>& nodes, ARGS... args)
	{
		return pack(nodes);
	}
};

template <typename T,egen::_GENERATED_OPCODE OPCODE>
struct ReducePacker
{
	virtual ~ReducePacker (void) = default;

	ArgsT<T> pack (const NodesT<T>& nodes, std::set<teq::RankT> dims)
	{
		if (std::any_of(dims.begin(), dims.end(),
			[](teq::RankT d) { return d >= teq::rank_cap; }))
		{
			logs::fatalf(
				"cannot reduce dimensions beyond rank cap %d", teq::rank_cap);
		}
		NodeptrT<T> node = nodes[0];
		teq::Shape shape = node->shape();
		std::vector<teq::DimT> slist(shape.begin(), shape.end());
		for (auto it = dims.begin(), et = dims.end(); it != et;)
		{
			if (slist.at(*it) > 1)
			{
				slist[*it] = 1;
				++it;
			}
			else
			{
				it = dims.erase(it);
			}
		}
		if (dims.empty())
		{
			// todo: warn
			return {};
		}
		std::vector<double> rdims(dims.begin(), dims.end());
		std::sort(rdims.begin(), rdims.end());
		return {Edge<T>(node, teq::Shape(slist), rdims)};
	}

	ArgsT<T> pack (const NodesT<T>& nodes, teq::RankT offset, teq::RankT ndims)
	{
		if (offset >= teq::rank_cap)
		{
			logs::fatalf("cannot reduce dimensions [%d:]. Must be less than %d",
				offset, teq::rank_cap);
		}
		std::vector<teq::RankT> dims(std::min(ndims, (teq::RankT) (teq::rank_cap - offset)));
		std::iota(dims.begin(), dims.end(), offset);
		return pack(nodes, std::set<teq::RankT>(dims.begin(), dims.end()));
	}
};

template <typename T>
struct FuncPacker<T,egen::REDUCE_SUM> final : private ReducePacker<T,egen::REDUCE_SUM>
{
	using ReducePacker<T,egen::REDUCE_SUM>::pack;

	ArgsT<T> pack (const NodesT<T>& nodes)
	{
		return {};
	}

	template <typename ...ARGS>
	ArgsT<T> pack (const NodesT<T>& nodes, ARGS... args)
	{
		return pack(nodes);
	}
};

template <typename T>
struct FuncPacker<T,egen::REDUCE_PROD> final : private ReducePacker<T,egen::REDUCE_PROD>
{
	using ReducePacker<T,egen::REDUCE_PROD>::pack;

	ArgsT<T> pack (const NodesT<T>& nodes)
	{
		return {};
	}

	template <typename ...ARGS>
	ArgsT<T> pack (const NodesT<T>& nodes, ARGS... args)
	{
		return pack(nodes);
	}
};

template <typename T>
struct FuncPacker<T,egen::REDUCE_MIN> final : private ReducePacker<T,egen::REDUCE_MIN>
{
	using ReducePacker<T,egen::REDUCE_MIN>::pack;

	ArgsT<T> pack (const NodesT<T>& nodes)
	{
		return {};
	}

	template <typename ...ARGS>
	ArgsT<T> pack (const NodesT<T>& nodes, ARGS... args)
	{
		return pack(nodes);
	}
};

template <typename T>
struct FuncPacker<T,egen::REDUCE_MAX> final : private ReducePacker<T,egen::REDUCE_MAX>
{
	using ReducePacker<T,egen::REDUCE_MAX>::pack;

	ArgsT<T> pack (const NodesT<T>& nodes)
	{
		return {};
	}

	template <typename ...ARGS>
	ArgsT<T> pack (const NodesT<T>& nodes, ARGS... args)
	{
		return pack(nodes);
	}
};

template <typename T>
struct FuncPacker<T,egen::ARGMAX> final
{
	ArgsT<T> pack (const NodesT<T>& nodes)
	{
		return {};
	}

	ArgsT<T> pack (const NodesT<T>& nodes, teq::RankT return_dim)
	{
		NodeptrT<T> node = nodes[0];
		teq::Shape shape = node->shape();
		if (shape.n_elems() == 1 ||
			(return_dim < teq::rank_cap && shape.at(return_dim) == 1))
		{
			return {};
		}

		std::vector<teq::DimT> slist;
		if (return_dim < teq::rank_cap)
		{
			slist = std::vector<teq::DimT>(shape.begin(), shape.end());
			slist[return_dim] = 1;
		}
		return {Edge<T>(node, teq::Shape(slist), {
			static_cast<double>(return_dim)})};
	}

	template <typename ...ARGS>
	ArgsT<T> pack (const NodesT<T>& nodes, ARGS... args)
	{
		return pack(nodes);
	}
};

template <typename T>
struct FuncPacker<T,egen::SLICE> final
{
	ArgsT<T> pack (const NodesT<T>& nodes)
	{
		return {};
	}

	ArgsT<T> pack (const NodesT<T>& nodes, const eigen::PairVecT<teq::DimT>& extents)
	{
		if (extents.size() > teq::rank_cap)
		{
			eigen::PairVecT<int> readable_extents(extents.begin(), extents.end());
			logs::fatalf(
				"cannot slice dimensions beyond rank_cap %d: using extent %s", teq::rank_cap,
				fmts::to_string(readable_extents.begin(), readable_extents.end()).c_str());
		}
		eteq::NodeptrT<T> arg = nodes[0];
		teq::Shape shape = arg->shape();
		std::vector<teq::DimT> slist(shape.begin(), shape.end());
		slist.reserve(teq::rank_cap);
		eigen::PairVecT<teq::DimT> xlist;
		xlist.reserve(extents.size());
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
			teq::DimT xtend = std::min(ex.second, (teq::DimT) (shape.at(i) - offset));
			slist[i] = xtend;
			xlist.push_back({offset, xtend});
		}
		return {eteq::Edge<T>(arg, teq::Shape(slist), eigen::encode_pair(xlist))};
	}

	template <typename ...ARGS>
	ArgsT<T> pack (const NodesT<T>& nodes, ARGS... args)
	{
		return pack(nodes);
	}
};

template <typename T>
struct FuncPacker<T,egen::PAD> final
{
	ArgsT<T> pack (const NodesT<T>& nodes)
	{
		return {};
	}

	ArgsT<T> pack (const NodesT<T>& nodes, const eigen::PairVecT<teq::DimT>& paddings)
	{
		if (paddings.size() > teq::rank_cap)
		{
			eigen::PairVecT<int> readable_paddings(paddings.begin(), paddings.end());
			logs::fatalf(
				"cannot pad dimensions beyond rank_cap %d: using paddings %s", teq::rank_cap,
				fmts::to_string(readable_paddings.begin(), readable_paddings.end()).c_str());
		}
		eteq::NodeptrT<T> arg = nodes[0];
		teq::Shape shape = arg->shape();
		std::vector<teq::DimT> slist(shape.begin(), shape.end());
		size_t n = std::min(paddings.size(), (size_t) teq::rank_cap);
		for (size_t i = 0; i < n; ++i)
		{
			slist[i] += paddings[i].first + paddings[i].second;
		}
		return {eteq::Edge<T>(arg, teq::Shape(slist), eigen::encode_pair(paddings))};
	}

	template <typename ...ARGS>
	ArgsT<T> pack (const NodesT<T>& nodes, ARGS... args)
	{
		return pack(nodes);
	}
};

template <typename T>
struct FuncPacker<T,egen::STRIDE> final
{
	ArgsT<T> pack (const NodesT<T>& nodes)
	{
		return {};
	}

	ArgsT<T> pack (const NodesT<T>& nodes, const std::vector<teq::DimT>& incrs)
	{
		if (incrs.size() > teq::rank_cap)
		{
			logs::warnf("trying to stride in dimensions beyond rank_cap %d: "
				"using increments %s (will ignore those dimensions)", teq::rank_cap,
				fmts::to_string(incrs.begin(), incrs.end()).c_str());
		}
		eteq::NodeptrT<T> arg = nodes[0];
		std::vector<double> coords(teq::rank_cap, 1);
		size_t n = std::min(incrs.size(), (size_t) teq::rank_cap);
		for (size_t i = 0; i < n; ++i)
		{
			coords[i] = incrs[i];
		}
		teq::Shape shape = arg->shape();
		std::vector<teq::DimT> slist(shape.begin(), shape.end());
		for (size_t i = 0; i < n; ++i)
		{
			slist[i] = std::round((double) slist[i] / incrs[i]);
		}
		return {eteq::Edge<T>(arg, teq::Shape(slist), coords)};
	}

	template <typename ...ARGS>
	ArgsT<T> pack (const NodesT<T>& nodes, ARGS... args)
	{
		return pack(nodes);
	}
};

template <typename T>
struct FuncPacker<T,egen::SCATTER> final
{
	ArgsT<T> pack (const NodesT<T>& nodes)
	{
		return {};
	}

	ArgsT<T> pack (const NodesT<T>& nodes, const teq::Shape outshape,
		const std::vector<teq::DimT>& incrs)
	{
		if (incrs.size() > teq::rank_cap)
		{
			logs::warnf("trying to scatter in dimensions beyond rank_cap %d: "
				"using increments %s (will ignore those dimensions)", teq::rank_cap,
				fmts::to_string(incrs.begin(), incrs.end()).c_str());
		}
		eteq::NodeptrT<T> arg = nodes[0];
		std::vector<double> coords(teq::rank_cap, 1);
		size_t n = std::min(incrs.size(), (size_t) teq::rank_cap);
		for (size_t i = 0; i < n; ++i)
		{
			coords[i] = incrs[i];
		}
		return {eteq::Edge<T>(arg, outshape, coords)};
	}

	template <typename ...ARGS>
	ArgsT<T> pack (const NodesT<T>& nodes, ARGS... args)
	{
		return pack(nodes);
	}
};

template <typename T>
struct FuncPacker<T,egen::MATMUL> final
{
	ArgsT<T> pack (const NodesT<T>& nodes)
	{
		return {};
	}

	ArgsT<T> pack (const NodesT<T>& nodes, const eigen::PairVecT<teq::RankT>& dims)
	{
		eteq::NodeptrT<T> a = nodes[0];
		eteq::NodeptrT<T> b = nodes[1];
		teq::Shape ashape = a->shape();
		teq::Shape bshape = b->shape();
		// check common dimensions
		std::array<bool,teq::rank_cap> avisit;
		std::array<bool,teq::rank_cap> bvisit;
		std::fill(avisit.begin(), avisit.end(), false);
		std::fill(bvisit.begin(), bvisit.end(), false);
		for (const std::pair<teq::RankT,teq::RankT>& coms : dims)
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
			eteq::Edge<T>(a, outshape, eigen::encode_pair(dims)),
			eteq::Edge<T>(b, outshape, {}),
		};
	}

	template <typename ...ARGS>
	ArgsT<T> pack (const NodesT<T>& nodes, ARGS... args)
	{
		return pack(nodes);
	}
};

template <typename T>
struct FuncPacker<T,egen::CONV> final
{
	ArgsT<T> pack (const NodesT<T>& nodes)
	{
		return {};
	}

	ArgsT<T> pack (const NodesT<T>& nodes, const std::vector<teq::RankT>& dims)
	{
		eteq::NodeptrT<T> image = nodes[0];
		eteq::NodeptrT<T> kernel = nodes[1];
		teq::Shape inshape = image->shape();
		teq::Shape kernelshape = kernel->shape();
		size_t n = std::min(dims.size(), (size_t) teq::rank_cap);
		if (std::any_of(kernelshape.begin() + n, kernelshape.end(),
			[](teq::DimT d) { return d > 1; }))
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
			eteq::Edge<T>(image, outshape, {}),
			eteq::Edge<T>(kernel, outshape, kernel_dims),
		};
	}

	template <typename ...ARGS>
	ArgsT<T> pack (const NodesT<T>& nodes, ARGS... args)
	{
		return pack(nodes);
	}
};

template <typename T>
struct FuncPacker<T,egen::REVERSE> final
{
	ArgsT<T> pack (const NodesT<T>& nodes)
	{
		return {};
	}

	ArgsT<T> pack (const NodesT<T>& nodes, const std::vector<teq::RankT>& dims)
	{
		eteq::NodeptrT<T> arg = nodes[0];
		return {eteq::Edge<T>(arg,arg->shape(),std::vector<double>(dims.begin(),dims.end()))};
	}

	template <typename ...ARGS>
	ArgsT<T> pack (const NodesT<T>& nodes, ARGS... args)
	{
		return pack(nodes);
	}
};

static bool is_inorder (const std::vector<teq::RankT>& order)
{
	size_t n = order.size();
	bool inorder = n > 0 ? (order[0] == 0) : true;
	for (size_t i = 1; i < n && inorder; ++i)
	{
		inorder = inorder && (order[i] == (order[i-1] + 1));
	}
	return inorder;
}

template <typename T>
struct FuncPacker<T,egen::PERMUTE> final
{
	ArgsT<T> pack (const NodesT<T>& nodes)
	{
		return {};
	}

	ArgsT<T> pack (const NodesT<T>& nodes, const std::vector<teq::RankT>& order)
	{
		if (is_inorder(order))
		{
			logs::warn("permuting with same dimensions ... treating as identity");
			return {};
		}
		eteq::NodeptrT<T> arg = nodes[0];
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
		std::vector<teq::RankT> indices = order;
		for (teq::RankT i = 0; i < teq::rank_cap; ++i)
		{
			if (false == visited[i])
			{
				indices.push_back(i);
			}
		}
		teq::Shape shape = arg->shape();
		std::vector<teq::DimT> slist(teq::rank_cap, 0);
		for (teq::RankT i = 0; i < teq::rank_cap; ++i)
		{
			slist[i] = shape.at(indices[i]);
		}
		return {
			eteq::Edge<T>(arg, teq::Shape(slist),
				std::vector<double>(indices.begin(), indices.end()))
		};
	}

	template <typename ...ARGS>
	ArgsT<T> pack (const NodesT<T>& nodes, ARGS... args)
	{
		return pack(nodes);
	}
};

template <typename T>
struct FuncPacker<T,egen::EXTEND> final
{
	ArgsT<T> pack (const NodesT<T>& nodes)
	{
		return {};
	}

	ArgsT<T> pack (const NodesT<T>& nodes, const std::vector<teq::DimT>& bcast)
	{
		if (bcast.empty() || std::all_of(bcast.begin(), bcast.end(),
			[](teq::DimT d) { return 1 == d; }))
		{
			logs::warn("extending with nothing... treating as identity");
			return {}; // identity
		}
		if (std::any_of(bcast.begin(), bcast.end(),
			[](teq::DimT d) { return 0 == d; }))
		{
			logs::fatalf("cannot extend using zero dimensions %s",
				fmts::to_string(bcast.begin(), bcast.end()).c_str());
		}
		size_t nbcasts = bcast.size();
		if (nbcasts > teq::rank_cap)
		{
			logs::fatalf("cannot extend shape ranks %s beyond rank_cap",
				fmts::to_string(bcast.begin(), bcast.end()).c_str());
		}
		while (nbcasts > 0 && 1 == bcast[nbcasts - 1])
		{
			--nbcasts;
		}
		eteq::NodeptrT<T> arg = nodes[0];
		teq::Shape shape = arg->shape();
		std::vector<teq::DimT> slist(shape.begin(), shape.end());
		for (size_t i = 0; i < nbcasts; ++i)
		{
			if (bcast.at(i) > 1 && shape.at(i) > 1)
			{
				logs::fatalf("cannot extend non-singular dimension %d of shape %s",
					i, shape.to_string().c_str());
			}
			slist[i] *= bcast[i];
		}
		return {eteq::Edge<T>(arg, teq::Shape(slist),
			std::vector<double>(bcast.begin(), bcast.begin() + nbcasts))};
	}

	ArgsT<T> pack (const NodesT<T>& nodes, teq::RankT offset, const std::vector<teq::DimT>& xlist)
	{
		std::vector<teq::DimT> bcast(offset, 1);
		bcast.insert(bcast.end(), xlist.begin(), xlist.end());
		return pack(nodes, bcast);
	}

	template <typename ...ARGS>
	ArgsT<T> pack (const NodesT<T>& nodes, ARGS... args)
	{
		return pack(nodes);
	}
};

template <typename T>
struct FuncPacker<T,egen::CONCAT> final
{
	ArgsT<T> pack (const NodesT<T>& nodes)
	{
		return {};
	}

	ArgsT<T> pack (const NodesT<T>& nodes, teq::RankT axis)
	{
		eteq::NodeptrT<T> left = nodes[0];
		eteq::NodeptrT<T> right = nodes[1];
		teq::Shape leftshape = left->shape();
		teq::Shape rightshape = right->shape();
		std::vector<teq::DimT> slist(leftshape.begin(), leftshape.end());
		slist[axis] += rightshape.at(axis);
		teq::Shape outshape(slist);
		return {
			eteq::Edge<T>(left, outshape, {static_cast<double>(axis)}),
			eteq::Edge<T>(right, outshape, {}),
		};
	}

	template <typename ...ARGS>
	ArgsT<T> pack (const NodesT<T>& nodes, ARGS... args)
	{
		return pack(nodes);
	}
};

template <typename T>
struct FuncPacker<T,egen::GROUP_CONCAT> final
{
	ArgsT<T> pack (const NodesT<T>& nodes)
	{
		return {};
	}

	ArgsT<T> pack (const NodesT<T>& nodes, teq::RankT axis)
	{
		if (nodes.size() == 1)
		{
			return {};
		}
		size_t nargs = nodes.size();
		if (nargs < 2)
		{
			logs::fatal("cannot group concat less than 2 arguments");
		}
		if (std::any_of(nodes.begin(), nodes.end(),
			[](eteq::NodeptrT<T> arg) { return nullptr == arg; }))
		{
			logs::fatal("cannot group concat with null argument");
		}
		teq::Shape initshape = nodes[0]->shape();
		std::vector<teq::DimT> slist(initshape.begin(), initshape.end());
		for (size_t i = 1; i < nargs; ++i)
		{
			slist[axis] += nodes[i]->shape().at(axis);
		}
		teq::Shape outshape(slist);
		eteq::ArgsT<T> groups;
		groups.reserve(nargs);
		groups.push_back(eteq::Edge<T>(nodes[0], outshape, {static_cast<double>(axis)}));
		std::transform(nodes.begin() + 1, nodes.end(), std::back_inserter(groups),
			[&](eteq::NodeptrT<T> arg) { return eteq::Edge<T>(arg, outshape, {}); });
		return groups;
	}

	template <typename ...ARGS>
	ArgsT<T> pack (const NodesT<T>& nodes, ARGS... args)
	{
		return pack(nodes);
	}
};

template <typename T>
struct FuncPacker<T,egen::RESHAPE> final
{
	ArgsT<T> pack (const NodesT<T>& nodes)
	{
		return {};
	}

	ArgsT<T> pack (const NodesT<T>& nodes, teq::Shape shape)
	{
		eteq::NodeptrT<T> arg = nodes[0];
		return {eteq::Edge<T>(arg, shape, {})};
	}

	template <typename ...ARGS>
	ArgsT<T> pack (const NodesT<T>& nodes, ARGS... args)
	{
		return pack(nodes);
	}
};

#define CHOOSE_PACK(OPCODE)args = FuncPacker<T,OPCODE>().pack(nodes, vargs...);

/// Return functor node given opcode and node arguments
template <typename T, typename ...ARGS>
NodeptrT<T> make_functor (egen::_GENERATED_OPCODE opcode, NodesT<T> nodes, ARGS... vargs)
{
	if (nodes.empty())
	{
		logs::fatalf("cannot %s without arguments", egen::name_op(opcode).c_str());
	}
	ArgsT<T> args;
	OPCODE_LOOKUP(CHOOSE_PACK, opcode)
	if (args.empty())
	{
		return nodes.front();
	}
	return std::make_shared<FunctorNode<T>>(
		std::shared_ptr<Functor<T>>(Functor<T>::get(opcode, args)));
}

}

#endif // ETEQ_FUNCTOR_HPP
