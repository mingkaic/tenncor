//
/// functor.hpp
/// eteq
///
/// Purpose:
/// Eigen functor implementation of operable func
///

#include "tag/locator.hpp"

#include "eteq/ifunctor.hpp"

#ifndef ETEQ_FUNCTOR_HPP
#define ETEQ_FUNCTOR_HPP

namespace eteq
{

/// Functor implementation of operable functor of Eigen operators
template <typename T>
struct Functor final : public iFunctor<T>
{
	/// Return Functor given opcodes mapped to Eigen operators in operator.hpp
	static Functor<T>* get (egen::_GENERATED_OPCODE opcode,
		LinksT<T> args, marsh::Maps&& attrs)
	{
		return new Functor<T>(opcode, args, std::move(attrs));
	}

	~Functor (void)
	{
		for (LinkptrT<T> arg : args_)
		{
			arg->unsubscribe(this);
		}
	}

	/// Return deep copy of this Functor
	Functor<T>* clone (void) const
	{
		return static_cast<Functor<T>*>(clone_impl());
	}

	Functor (Functor<T>&& other) = delete;

	Functor<T>& operator = (const Functor<T>& other) = delete;

	Functor<T>& operator = (Functor<T>&& other) = delete;

	/// Implementation of iTensor
	teq::Shape shape (void) const override
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
	teq::EdgeRefsT get_children (void) const override
	{
		teq::EdgeRefsT refs;
		refs.reserve(args_.size());
		std::transform(args_.begin(), args_.end(), std::back_inserter(refs),
			[](LinkptrT<T> edge) -> const teq::iEdge&
			{
				return *edge;
			});
		return refs;
	}

	/// Implementation of iFunctor
	marsh::iObject* get_attr (std::string attr_name) const override
	{
		return estd::has(attrs_.contents_, attr_name) ?
			attrs_.contents_.at(attr_name).get() : nullptr;
	}

	/// Implementation of iFunctor
	std::vector<std::string> ls_attrs (void) const override
	{
		return attrs_.keys();
	}

	/// Implementation of iFunctor
	void update_child (teq::TensptrT arg, size_t index) override
	{
		if (index >= args_.size())
		{
			logs::fatalf("cannot modify argument %d "
				"when there are only %d arguments",
				index, args_.size());
		}
		auto link = static_cast<iLink<T>*>(args_[index].get());
		if (arg != link->get_tensor())
		{
			uninitialize();
			link->unsubscribe(this);
			teq::Shape nexshape = arg->shape();
			teq::Shape curshape = link->shape();
			if (false == nexshape.compatible_after(curshape, 0))
			{
				logs::fatalf("cannot update child %d to argument with "
					"incompatible shape %s (requires shape %s)",
					index, nexshape.to_string().c_str(),
					curshape.to_string().c_str());
			}
			args_[index] = to_node<T>(arg);
			link->subscribe(this);
		}
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

	/// Implementation of iFunctor<T>
	bool is_uninit (void) const override
	{
		return nullptr == out_;
	}

	/// Removes internal Eigen data object
	void uninitialize (void)
	{
		if (is_uninit())
		{
			return;
		}
		for (auto& parent : this->subs_)
		{
			parent->uninitialize();
		}
		out_ = nullptr;
	}

	/// Populate internal Eigen data object
	void initialize (void)
	{
		eigen::EEdgeRefsT<T> refs;
		refs.reserve(args_.size());
		std::transform(args_.begin(), args_.end(), std::back_inserter(refs),
			[](LinkptrT<T> edge) -> const eigen::iEigenEdge<T>&
			{
				return *edge;
			});
		egen::typed_exec<T>((egen::_GENERATED_OPCODE) opcode_.code_,
			out_, shape_, refs, attrs_);
	}

private:
	Functor (egen::_GENERATED_OPCODE opcode,
		LinksT<T> args, marsh::Maps&& attrs) :
		opcode_(teq::Opcode{egen::name_op(opcode), opcode}),
		args_(args.begin(), args.end()),
		attrs_(std::move(attrs))
	{
		if (args.empty())
		{
			logs::fatalf("cannot perform `%s` without arguments",
				egen::name_op(opcode).c_str());
		}

		auto shape_attr = get_attr(eigen::shaper_key);
		if (marsh::NumArray<double>* arr = nullptr == shape_attr ?
			nullptr : shape_attr->template cast<marsh::NumArray<double>>())
		{
			std::vector<teq::DimT> slist(
				arr->contents_.begin(), arr->contents_.end());
			shape_ = teq::Shape(slist);
		}
		else
		{
			shape_ = args_.front()->shape();
		}
		common_init();
	}

	Functor (const Functor<T>& other) :
		opcode_(other.opcode_),
		shape_(other.shape_),
		args_(other.args_)
	{
		std::unique_ptr<marsh::Maps> mattr(other.attrs_.clone());
		attrs_ = std::move(*mattr);
		common_init();
	}

	teq::iTensor* clone_impl (void) const override
	{
		return new Functor<T>(*this);
	}

	void common_init (void)
	{
		for (LinkptrT<T> arg : args_)
		{
			arg->subscribe(this);
		}
#ifndef SKIP_INIT
		if (std::all_of(args_.begin(), args_.end(),
			[](LinkptrT<T>& link)
			{
				return link->has_data();
			}))
		{
			initialize();
		}
#endif // SKIP_INIT
	}

	eigen::EigenptrT<T> out_ = nullptr;

	/// Operation encoding
	teq::Opcode opcode_;

	/// Shape info built at construction time according to arguments
	teq::Shape shape_;

	/// Tensor arguments (and children)
	LinksT<T> args_;

	marsh::Maps attrs_;
};

// todo: move these to eigen and auto-generate
template <typename T,egen::_GENERATED_OPCODE OPCODE>
struct FuncPacker final
{
	bool pack (marsh::Maps& attrs, const LinksT<T>& nodes)
	{
		return true;
	}

	template <typename ...ARGS>
	bool pack (marsh::Maps& attrs, const LinksT<T>& nodes, ARGS... args)
	{
		return pack(attrs, nodes);
	}
};

template <typename T>
struct EmptyPacker
{
	virtual ~EmptyPacker (void) = default;

	bool pack (marsh::Maps& attrs, const LinksT<T>& nodes)
	{
		return false;
	}

	template <typename ...ARGS>
	bool pack (marsh::Maps& attrs, const LinksT<T>& nodes, ARGS... args)
	{
		return pack(attrs, nodes);
	}
};

template <typename T,egen::_GENERATED_OPCODE OPCODE>
struct ReducePacker : private EmptyPacker<T>
{
	virtual ~ReducePacker (void) = default;

	using EmptyPacker<T>::pack;

	bool pack (marsh::Maps& attrs, const LinksT<T>& nodes, std::set<teq::RankT> dims)
	{
		if (std::any_of(dims.begin(), dims.end(),
			[](teq::RankT d) { return d >= teq::rank_cap; }))
		{
			logs::fatalf(
				"cannot reduce dimensions beyond rank cap %d", teq::rank_cap);
		}
		LinkptrT<T> node = nodes.front();
		teq::Shape shape = node->shape();
		std::vector<teq::DimT> slist(shape.begin(), shape.end());
		std::set<teq::RankT> sig_dims = dims;
		for (auto it = sig_dims.begin(), et = sig_dims.end(); it != et;)
		{
			if (slist.at(*it) > 1)
			{
				slist[*it] = 1;
				++it;
			}
			else
			{
				it = sig_dims.erase(it);
			}
		}
		if (sig_dims.empty())
		{
			logs::debugf("reducing with no significant dimensions... "
				"treating as identity: (dims=%s, shape=%s)",
				fmts::to_string(dims.begin(), dims.end()).c_str(),
				shape.to_string().c_str());
			return false;
		}
		std::vector<double> rslist(slist.begin(), slist.end());
		attrs.contents_.emplace(eigen::shaper_key,
			std::make_unique<marsh::NumArray<double>>(rslist));
		std::vector<double> rdims(sig_dims.begin(), sig_dims.end());
		std::sort(rdims.begin(), rdims.end());
		attrs.contents_.emplace(eigen::coorder_key,
			std::make_unique<marsh::NumArray<double>>(rdims));
		return true;
	}

	bool pack (marsh::Maps& attrs, const LinksT<T>& nodes, teq::RankT offset, teq::RankT ndims)
	{
		if (offset >= teq::rank_cap)
		{
			logs::fatalf("cannot reduce dimensions [%d:]. Must be less than %d",
				offset, teq::rank_cap);
		}
		std::vector<teq::RankT> dims(std::min(ndims, (teq::RankT) (teq::rank_cap - offset)));
		std::iota(dims.begin(), dims.end(), offset);
		return pack(attrs, nodes, std::set<teq::RankT>(dims.begin(), dims.end()));
	}
};

template <typename T>
struct FuncPacker<T,egen::REDUCE_SUM> final : private ReducePacker<T,egen::REDUCE_SUM>
{
	using ReducePacker<T,egen::REDUCE_SUM>::pack;
};

template <typename T>
struct FuncPacker<T,egen::REDUCE_PROD> final : private ReducePacker<T,egen::REDUCE_PROD>
{
	using ReducePacker<T,egen::REDUCE_PROD>::pack;
};

template <typename T>
struct FuncPacker<T,egen::REDUCE_MIN> final : private ReducePacker<T,egen::REDUCE_MIN>
{
	using ReducePacker<T,egen::REDUCE_MIN>::pack;
};

template <typename T>
struct FuncPacker<T,egen::REDUCE_MAX> final : private ReducePacker<T,egen::REDUCE_MAX>
{
	using ReducePacker<T,egen::REDUCE_MAX>::pack;
};

template <typename T>
struct FuncPacker<T,egen::ARGMAX> final : private EmptyPacker<T>
{
	using EmptyPacker<T>::pack;

	bool pack (marsh::Maps& attrs, const LinksT<T>& nodes, teq::RankT return_dim)
	{
		LinkptrT<T> node = nodes.front();
		teq::Shape shape = node->shape();
		if (shape.n_elems() == 1 ||
			(return_dim < teq::rank_cap && shape.at(return_dim) == 1))
		{
			logs::debugf("argreducing with no significant dimensions... "
				"treating as identity: (return_dim=%d, shape=%s)",
				(int) return_dim, shape.to_string().c_str());
			return false;
		}

		std::vector<teq::DimT> slist;
		if (return_dim < teq::rank_cap)
		{
			slist = std::vector<teq::DimT>(shape.begin(), shape.end());
			slist[return_dim] = 1;
		}
		std::vector<double> rslist(slist.begin(), slist.end());
		attrs.contents_.emplace(eigen::shaper_key,
			std::make_unique<marsh::NumArray<double>>(rslist));
		attrs.contents_.emplace(eigen::coorder_key,
			std::make_unique<marsh::NumArray<double>>(
				std::vector<double>{(double) return_dim}));
		return true;
	}
};

template <typename T>
struct FuncPacker<T,egen::SLICE> final : private EmptyPacker<T>
{
	using EmptyPacker<T>::pack;

	bool pack (marsh::Maps& attrs, const LinksT<T>& nodes, const eigen::PairVecT<teq::DimT>& extents)
	{
		if (extents.size() > teq::rank_cap)
		{
			logs::fatalf(
				"cannot slice dimensions beyond rank_cap %d: using extent %s",
				teq::rank_cap, eigen::to_string(extents).c_str());
		}
		LinkptrT<T> arg = nodes.front();
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
				logs::fatalf("cannot extend zero slices: extents %s",
					eigen::to_string(extents).c_str());
			}
			teq::DimT offset = std::min(ex.first, (teq::DimT) (shape.at(i) - 1));
			teq::DimT xtend = std::min(ex.second, (teq::DimT) (shape.at(i) - offset));
			slist[i] = xtend;
			xlist.push_back({offset, xtend});
		}
		teq::Shape outshape(slist);
		if (outshape.compatible_after(shape, 0))
		{
			logs::debugf("slice parameter covers whole tensor... "
				"treating as identity: (extents=%s)",
				eigen::to_string(extents).c_str());
			return false;
		}
		std::vector<double> rslist(slist.begin(), slist.end());
		attrs.contents_.emplace(eigen::shaper_key,
			std::make_unique<marsh::NumArray<double>>(rslist));
		attrs.contents_.emplace(eigen::coorder_key,
			std::make_unique<marsh::NumArray<double>>(
				eigen::encode_pair(xlist)));
		return true;
	}
};

template <typename T>
struct FuncPacker<T,egen::PAD> final : private EmptyPacker<T>
{
	using EmptyPacker<T>::pack;

	bool pack (marsh::Maps& attrs, const LinksT<T>& nodes, const eigen::PairVecT<teq::DimT>& paddings)
	{
		if (paddings.size() > teq::rank_cap)
		{
			logs::fatalf(
				"cannot pad dimensions beyond rank_cap %d: using paddings %s",
				teq::rank_cap, eigen::to_string(paddings).c_str());
		}
		LinkptrT<T> arg = nodes.front();
		teq::Shape shape = arg->shape();
		std::vector<teq::DimT> slist(shape.begin(), shape.end());
		size_t n = std::min(paddings.size(), (size_t) teq::rank_cap);
		for (size_t i = 0; i < n; ++i)
		{
			slist[i] += paddings[i].first + paddings[i].second;
		}
		std::vector<double> rslist(slist.begin(), slist.end());
		attrs.contents_.emplace(eigen::shaper_key,
			std::make_unique<marsh::NumArray<double>>(rslist));
		attrs.contents_.emplace(eigen::coorder_key,
			std::make_unique<marsh::NumArray<double>>(
				eigen::encode_pair(paddings)));
		return true;
	}
};

template <typename T>
struct FuncPacker<T,egen::STRIDE> final : private EmptyPacker<T>
{
	using EmptyPacker<T>::pack;

	bool pack (marsh::Maps& attrs, const LinksT<T>& nodes, const std::vector<teq::DimT>& incrs)
	{
		if (incrs.size() > teq::rank_cap)
		{
			logs::warnf("trying to stride in dimensions beyond rank_cap %d: "
				"using increments %s (will ignore those dimensions)", teq::rank_cap,
				fmts::to_string(incrs.begin(), incrs.end()).c_str());
		}
		LinkptrT<T> arg = nodes.front();
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
		std::vector<double> rslist(slist.begin(), slist.end());
		attrs.contents_.emplace(eigen::shaper_key,
			std::make_unique<marsh::NumArray<double>>(rslist));
		attrs.contents_.emplace(eigen::coorder_key,
			std::make_unique<marsh::NumArray<double>>(coords));
		return true;
	}
};

template <typename T>
struct FuncPacker<T,egen::SCATTER> final : private EmptyPacker<T>
{
	using EmptyPacker<T>::pack;

	bool pack (marsh::Maps& attrs, const LinksT<T>& nodes, const teq::Shape outshape,
		const std::vector<teq::DimT>& incrs)
	{
		if (incrs.size() > teq::rank_cap)
		{
			logs::warnf("trying to scatter in dimensions beyond rank_cap %d: "
				"using increments %s (will ignore those dimensions)", teq::rank_cap,
				fmts::to_string(incrs.begin(), incrs.end()).c_str());
		}
		LinkptrT<T> arg = nodes.front();
		std::vector<double> coords(teq::rank_cap, 1);
		size_t n = std::min(incrs.size(), (size_t) teq::rank_cap);
		for (size_t i = 0; i < n; ++i)
		{
			coords[i] = incrs[i];
		}
		std::vector<double> rslist(outshape.begin(), outshape.end());
		attrs.contents_.emplace(eigen::shaper_key,
			std::make_unique<marsh::NumArray<double>>(rslist));
		attrs.contents_.emplace(eigen::coorder_key,
			std::make_unique<marsh::NumArray<double>>(coords));
		return true;
	}
};

template <typename T>
struct FuncPacker<T,egen::MATMUL> final : private EmptyPacker<T>
{
	using EmptyPacker<T>::pack;

	bool pack (marsh::Maps& attrs, const LinksT<T>& nodes, const eigen::PairVecT<teq::RankT>& dims)
	{
		LinkptrT<T> a = nodes[0];
		LinkptrT<T> b = nodes[1];
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
				logs::fatalf("invalid shapes %s and %s do not match "
					"common dimensions %s", ashape.to_string().c_str(),
					bshape.to_string().c_str(),
					eigen::to_string(dims).c_str());
			}
			if (avisit[coms.first] || bvisit[coms.second])
			{
				logs::fatalf("contraction dimensions %s must be unique for "
					"each side", eigen::to_string(dims).c_str());
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
		std::vector<double> rslist(outlist.begin(), outlist.end());
		attrs.contents_.emplace(eigen::shaper_key,
			std::make_unique<marsh::NumArray<double>>(rslist));
		attrs.contents_.emplace(eigen::coorder_key,
			std::make_unique<marsh::NumArray<double>>(
				eigen::encode_pair(dims)));
		return true;
	}
};

template <typename T>
struct FuncPacker<T,egen::CONV> final : private EmptyPacker<T>
{
	using EmptyPacker<T>::pack;

	bool pack (marsh::Maps& attrs, const LinksT<T>& nodes, const std::vector<teq::RankT>& dims)
	{
		LinkptrT<T> image = nodes[0];
		LinkptrT<T> kernel = nodes[1];
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
		std::vector<double> rslist(slist.begin(), slist.end());
		attrs.contents_.emplace(eigen::shaper_key,
			std::make_unique<marsh::NumArray<double>>(rslist));
		attrs.contents_.emplace(eigen::coorder_key,
			std::make_unique<marsh::NumArray<double>>(
				std::vector<double>(dims.begin(), dims.end())));
		return true;
	}
};

template <typename T>
struct FuncPacker<T,egen::REVERSE> final : private EmptyPacker<T>
{
	using EmptyPacker<T>::pack;

	bool pack (marsh::Maps& attrs, const LinksT<T>& nodes, const std::vector<teq::RankT>& dims)
	{
		attrs.contents_.emplace(eigen::coorder_key,
			std::make_unique<marsh::NumArray<double>>(
				std::vector<double>(dims.begin(),dims.end())));
		return true;
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

template <typename T>
struct FuncPacker<T,egen::PERMUTE> final : private EmptyPacker<T>
{
	using EmptyPacker<T>::pack;

	bool pack (marsh::Maps& attrs, const LinksT<T>& nodes, const std::vector<teq::RankT>& order)
	{
		if (is_inorder(order))
		{
			logs::debug("permuting with same dimensions ... treating as identity");
			return false;
		}
		LinkptrT<T> arg = nodes.front();
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
		std::vector<double> rslist(slist.begin(), slist.end());
		attrs.contents_.emplace(eigen::shaper_key,
			std::make_unique<marsh::NumArray<double>>(rslist));
		attrs.contents_.emplace(eigen::coorder_key,
			std::make_unique<marsh::NumArray<double>>(
				std::vector<double>(indices.begin(),indices.end())));
		return true;
	}
};

template <typename T>
struct FuncPacker<T,egen::EXTEND> final : private EmptyPacker<T>
{
	using EmptyPacker<T>::pack;

	bool pack (marsh::Maps& attrs, const LinksT<T>& nodes, const std::vector<teq::DimT>& bcast)
	{
		if (bcast.empty() || std::all_of(bcast.begin(), bcast.end(),
			[](teq::DimT d) { return 1 == d; }))
		{
			logs::debug("extending with nothing... treating as identity");
			return false;
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
		LinkptrT<T> arg = nodes.front();
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
		std::vector<double> rslist(slist.begin(), slist.end());
		attrs.contents_.emplace(eigen::shaper_key,
			std::make_unique<marsh::NumArray<double>>(rslist));
		attrs.contents_.emplace(eigen::coorder_key,
			std::make_unique<marsh::NumArray<double>>(std::vector<double>(
				bcast.begin(), bcast.begin() + nbcasts)));
		return true;
	}

	bool pack (marsh::Maps& attrs, const LinksT<T>& nodes, teq::RankT offset, const std::vector<teq::DimT>& xlist)
	{
		std::vector<teq::DimT> bcast(offset, 1);
		bcast.insert(bcast.end(), xlist.begin(), xlist.end());
		return pack(attrs, nodes, bcast);
	}
};

template <typename T>
struct FuncPacker<T,egen::CONCAT> final : private EmptyPacker<T>
{
	using EmptyPacker<T>::pack;

	bool pack (marsh::Maps& attrs, const LinksT<T>& nodes, teq::RankT axis)
	{
		LinkptrT<T> left = nodes[0];
		LinkptrT<T> right = nodes[1];
		teq::Shape leftshape = left->shape();
		teq::Shape rightshape = right->shape();
		std::vector<teq::DimT> slist(leftshape.begin(), leftshape.end());
		slist[axis] += rightshape.at(axis);
		teq::Shape outshape(slist);
		std::vector<double> rslist(slist.begin(), slist.end());
		attrs.contents_.emplace(eigen::shaper_key,
			std::make_unique<marsh::NumArray<double>>(rslist));
		attrs.contents_.emplace(eigen::coorder_key,
			std::make_unique<marsh::NumArray<double>>(
				std::vector<double>{(double) axis}));
		return true;
	}
};

template <typename T>
struct FuncPacker<T,egen::GROUP_CONCAT> final : private EmptyPacker<T>
{
	using EmptyPacker<T>::pack;

	bool pack (marsh::Maps& attrs, const LinksT<T>& nodes, teq::RankT axis)
	{
		if (nodes.size() == 1)
		{
			logs::debug("concatenating a single node... treating as identity");
			return false;
		}
		size_t nargs = nodes.size();
		if (nargs < 2)
		{
			logs::fatal("cannot group concat less than 2 arguments");
		}
		if (std::any_of(nodes.begin(), nodes.end(),
			[](LinkptrT<T> arg) { return nullptr == arg; }))
		{
			logs::fatal("cannot group concat with null argument");
		}
		if (std::any_of(nodes.begin(), nodes.end(),
			[axis](LinkptrT<T> arg) { return arg->shape().at(axis) > 1; }))
		{
			logs::fatal("cannot group concat nodes with dimension at axis greater than 1");
		}
		teq::Shape initshape = nodes[0]->shape();
		std::vector<teq::DimT> slist(initshape.begin(), initshape.end());
		slist[axis] = nargs;
		teq::Shape outshape(slist);
		std::vector<double> rslist(slist.begin(), slist.end());
		attrs.contents_.emplace(eigen::shaper_key,
			std::make_unique<marsh::NumArray<double>>(rslist));
		attrs.contents_.emplace(eigen::coorder_key,
			std::make_unique<marsh::NumArray<double>>(
				std::vector<double>{(double) axis}));
		return true;
	}
};

template <typename T>
struct FuncPacker<T,egen::RESHAPE> final : private EmptyPacker<T>
{
	using EmptyPacker<T>::pack;

	bool pack (marsh::Maps& attrs, const LinksT<T>& nodes, teq::Shape shape)
	{
		std::vector<double> slist(shape.begin(), shape.end());
		attrs.contents_.emplace(eigen::shaper_key,
			std::make_unique<marsh::NumArray<double>>(slist));
		return true;
	}
};

#define CHOOSE_PACK(OPCODE)if (false == FuncPacker<T,OPCODE>().pack(\
	attrs, nodes, vargs...)) { return nodes.front(); }

/// Return functor node given opcode and node arguments
template <typename T, typename ...ARGS>
LinkptrT<T> make_functor (egen::_GENERATED_OPCODE opcode, LinksT<T> nodes, ARGS... vargs)
{
	if (nodes.empty())
	{
		logs::fatalf("cannot %s without arguments", egen::name_op(opcode).c_str());
	}
	marsh::Maps attrs;
	OPCODE_LOOKUP(CHOOSE_PACK, opcode)
	return func_link<T>(teq::TensptrT(Functor<T>::get(opcode,
			nodes, std::move(attrs))));
}

#undef CHOOSE_PACK

}

#endif // ETEQ_FUNCTOR_HPP
