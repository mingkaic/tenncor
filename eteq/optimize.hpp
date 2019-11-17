/// parse.hpp
/// eteq
///
/// Purpose:
/// Extend optimization module by defining ETEQ node parsing
///

#include "opt/parse.hpp"
#include "opt/filter.hpp"

#include "eteq/generated/api.hpp"
#include "eteq/constant.hpp"
#include "eteq/functor.hpp"

#ifndef ETEQ_OPT_HPP
#define ETEQ_OPT_HPP

namespace eteq
{

template <typename T>
struct AnyTarget final : public opt::iTarget
{
	AnyTarget (std::string symbol) : symbol_(symbol) {}

	teq::TensptrT convert (
		teq::Shape outshape, const opt::Candidate& candidate) const override
	{
		return estd::must_getf(candidate.anys_, symbol_,
			"cannot find any symbol %s", symbol_.c_str());
	}

	std::string symbol_;
};

template <typename T>
struct ScalarTarget final : public opt::iTarget
{
	ScalarTarget (double scalar) : scalar_(scalar) {}

	teq::TensptrT convert (
		teq::Shape outshape, const opt::Candidate& candidate) const override
	{
		return make_constant_scalar(scalar_, outshape)->get_tensor();
	}

	T scalar_;
};

struct FTargEdge final
{
	FTargEdge (opt::TargptrT target, const ::PtrList& attrs) : target_(target)
	{
		if (::KV_PAIR != attrs.type_)
		{
			logs::fatalf("passing attributes by %d typed list", attrs.type_);
		}
		for (auto it = attrs.head_; nullptr != it; it = it->next_)
		{
			auto kv = (::KeyVal*) it->val_;
			std::string key(kv->key_);
			std::vector<double> values;
			for (auto jt = kv->val_.head_; nullptr != jt; jt = jt->next_)
			{
				values.push_back(jt->val_);
			}
			if (key == eigen::shaper_key)
			{
				shape_ = teq::Shape(std::vector<teq::DimT>(values.begin(), values.end()));
			}
			else if (key == eigen::coorder_key)
			{
				coords_ = values;
			}
		}
	}

	opt::TargptrT target_;

	teq::Shape shape_;

	std::vector<double> coords_;
};

template <typename T>
struct FuncTarget final : public opt::iTarget
{
	FuncTarget (std::string opname, std::vector<FTargEdge> args,
		std::string variadic) :
		opcode_(egen::get_op(opname)),
		args_(args), variadic_(variadic) {}

	teq::TensptrT convert (
		teq::Shape outshape, const opt::Candidate& candidate) const override
	{
		ArgsT<T> args;
		for (auto& targ : args_)
		{
			args.push_back(Edge<T>(
				to_node<T>(targ.target_->convert(outshape, candidate)),
				targ.shape_,
				targ.coords_
			));
		}
		if (variadic_.size() > 0)
		{
			auto& edges = candidate.variadic_.at(variadic_);
			for (const teq::iEdge& edge : edges)
			{
				args.push_back(*static_cast<const eteq::Edge<T>*>(&edge));
			}
		}
		return teq::TensptrT(eteq::Functor<T>::get(opcode_, args));
	}

	egen::_GENERATED_OPCODE opcode_;

	std::vector<FTargEdge> args_;

	std::string variadic_;
};

template <typename T>
opt::TargptrT build_target (::TreeNode* target)
{
	opt::TargptrT out;
	switch (target->type_)
	{
		case ::TreeNode::ANY:
			out = std::make_shared<AnyTarget<T>>(std::string(target->val_.any_));
			break;
		case ::TreeNode::SCALAR:
			out = std::make_shared<ScalarTarget<T>>(target->val_.scalar_);
			break;
		case ::TreeNode::FUNCTOR:
		{
			::Functor* func = target->val_.functor_;
			std::vector<FTargEdge> args;
			for (auto it = func->args_.head_; it != nullptr; it = it->next_)
			{
				auto arg = (::Arg*) it->val_;
				args.push_back(FTargEdge(
					build_target<T>(arg->node_), arg->attrs_));
			}
			out = std::make_shared<FuncTarget<T>>(std::string(func->name_),
				args, std::string(func->variadic_));
		}
			break;
		default:
			logs::fatalf("building unknown target %d", target->type_);
	}
	return out;
}

/// Return optimization rules tied to ETEQ Builder specified in content
template <typename T>
opt::CversionCtx parse (std::string content)
{
	return opt::parse(content, build_target<T>);
}

/// Return optimization rules tied to ETEQ Builder specified in file
template <typename T>
opt::CversionCtx parse_file (std::string filename)
{
	return opt::parse_file(filename, build_target<T>);
}

template <typename T>
struct Hasher final : public teq::OnceTraveler
{
	Hasher (tag::PropertyRegistry& prop_reg = tag::get_property_reg()) :
		prop_reg_(prop_reg) {}

	/// Implementation of OnceTraveler
	void visit_leaf (teq::iLeaf* leaf) override
	{
		if (leaf->is_const())
		{
			T* data = (T*) leaf->data();
			encode_label(leaf, fmts::to_string(data, data + leaf->shape().n_elems()));
		}
		else
		{
			encode_label(leaf, fmts::to_string((size_t) leaf));
		}
	}

	/// Implementation of OnceTraveler
	void visit_func (teq::iFunctor* func) override
	{
		auto children = func->get_children();
		std::vector<std::string> hshs;
		hshs.reserve(children.size());
		for (const teq::iEdge& child : children)
		{
			auto ctens = child.get_tensor();
			ctens->accept(*this);
			hshs.push_back(boost::uuids::to_string(hashes_.at(ctens.get())));
		}
		if (prop_reg_.has_property(func, tag::immutable_tag))
		{
			std::sort(hshs.begin(), hshs.end());
		}
		encode_label(func, func->get_opcode().name_ + "\\" +
			fmts::to_string(hshs.begin(), hshs.end()));
	}

	void encode_label (teq::iTensor* tens, const std::string& label)
	{
		boost::uuids::uuid uuid;
		if (false == estd::get(uuid, uuids_, label))
		{
			uuid = uuid_gen_();
			uuids_.emplace(label, uuid);
		}
		hashes_.emplace(tens, uuid);
	}

	std::unordered_map<teq::iTensor*,boost::uuids::uuid> hashes_;

	tag::PropertyRegistry& prop_reg_;

private:
	std::unordered_map<std::string,boost::uuids::uuid> uuids_;

	boost::uuids::random_generator uuid_gen_;
};

template <typename T>
void remove_duplicates (teq::TensptrsT& roots)
{
	Hasher<T> hasher;
	for (auto& root : roots)
	{
		root->accept(hasher);
	}
	opt::remove_duplicates(roots,
		[&](teq::TensptrT a, teq::TensptrT b)
		{
			return hasher.hashes_.at(a.get()) == hasher.hashes_.at(b.get());
		});
}

template <typename T>
teq::TensptrT constant_func (teq::FuncptrT& func, opt::ParentReplF replacer)
{
	return opt::constant_func(func, replacer,
		[](teq::FuncptrT func) -> teq::TensptrT
		{
			teq::Session sess;
			sess.track({func});
			sess.update_target({func.get()});
			T* data = (T*) static_cast<Functor<T>*>(func.get())->data();
			return make_constant(data, func->shape())->get_tensor();
		});
}

template <typename T>
void constant_funcs (teq::TensptrsT& roots)
{
	teq::Session sess;
	sess.track(roots);
	opt::constant_funcs(roots,
		[&sess](teq::FuncptrT func) -> teq::TensptrT
		{
			sess.update_target({func.get()});
			T* data = (T*) static_cast<Functor<T>*>(func.get())->data();
			return make_constant(data, func->shape())->get_tensor();
		});
}

template <typename T>
teq::TensptrsT optimize (teq::TensptrsT roots,
	const opt::CversionCtx& opts)
{
	opt::CustomFilters filters;
	filters.prefilters_.push_back(remove_duplicates<T>);
	filters.prefilters_.push_back(constant_funcs<T>);
	filters.prenode_filters_.push_back(constant_func<T>);
	filters.postfilters_.push_back(remove_duplicates<T>);
	return opt::optimize(roots, opts, filters);
}

/// Apply optimization to graph roots tracked by session
template <typename T>
void optimize (teq::iSession& sess, const opt::CversionCtx& opts)
{
	teq::TensptrSetT tracked_set = sess.get_tracked();
	teq::TensptrsT tracked(tracked_set.begin(), tracked_set.end());
	optimize<T>(tracked, opts);
	sess.clear();
	sess.track(tracked);
}

}

#endif // ETEQ_OPT_HPP
