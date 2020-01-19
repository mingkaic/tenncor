/// parse.hpp
/// eteq
///
/// Purpose:
/// Extend optimization module by defining ETEQ node parsing
///

#include "opt/parse.hpp"
#include "opt/filter.hpp"

#include "eteq/generated/api.hpp"
#include "eteq/make.hpp"

#ifndef ETEQ_OPT_HPP
#define ETEQ_OPT_HPP

namespace eteq
{

template <typename T>
struct AnyTarget final : public opt::iTarget
{
	AnyTarget (std::string symbol) : symbol_(symbol) {}

	teq::TensptrT convert (teq::Shape outshape,
		const opt::Candidate& candidate) const override
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

	teq::TensptrT convert (teq::Shape outshape,
		const opt::Candidate& candidate) const override
	{
		return make_constant_scalar(scalar_, outshape);
	}

	T scalar_;
};

template <typename T>
struct FuncTarget final : public opt::iTarget
{
	FuncTarget (std::string opname, std::vector<opt::TargptrT> args,
		std::string variadic, const ::PtrList& attrs) :
		opcode_(egen::get_op(opname)),
		args_(args), variadic_(variadic)
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
			attrs_.add_attr(key, std::make_unique<marsh::NumArray<double>>(values));
		}
	}

	teq::TensptrT convert (
		teq::Shape outshape,
		const opt::Candidate& candidate) const override
	{
		teq::TensptrsT children;
		children.reserve(args_.size());
		for (opt::TargptrT targ : args_)
		{
			// todo: reverse outshape
			children.push_back(targ->convert(outshape, candidate));
		}
		if (variadic_.size() > 0)
		{
			auto& args = candidate.variadic_.at(variadic_);
			children.insert(children.end(), args.begin(), args.end());
		}
		std::unique_ptr<marsh::Maps> attrs(attrs_.clone());
		return teq::TensptrT(Functor<T>::get(
			opcode_, children, std::move(*attrs)));
	}

	egen::_GENERATED_OPCODE opcode_;

	std::vector<opt::TargptrT> args_;

	std::string variadic_;

	marsh::Maps attrs_;
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
			std::vector<opt::TargptrT> args;
			for (auto it = func->args_.head_; it != nullptr; it = it->next_)
			{
				args.push_back(build_target<T>((::TreeNode*) it->val_));
			}
			out = std::make_shared<FuncTarget<T>>(std::string(func->name_),
				args, std::string(func->variadic_), func->attrs_);
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
struct Hasher final : public teq::iOnceTraveler
{
	teq::TensMapT<boost::uuids::uuid> hashes_;

private:
	/// Implementation of iOnceTraveler
	void visit_leaf (teq::iLeaf& leaf) override
	{
		if (teq::IMMUTABLE == leaf.get_usage())
		{
			std::string label = leaf.shape().to_string() + "|";
			T* data = (T*) leaf.data();
			label += fmts::to_string(data, data + leaf.shape().n_elems());
			encode_label(&leaf, label);
		}
		else
		{
			hashes_.emplace(&leaf, uuid_gen_());
		}
	}

	/// Implementation of iOnceTraveler
	void visit_func (teq::iFunctor& func) override
	{
		auto children = func.get_children();
		std::vector<std::string> hshs;
		hshs.reserve(children.size());
		for (teq::TensptrT child : children)
		{
			child->accept(*this);
			hshs.push_back(boost::uuids::to_string(hashes_.at(child.get())));
		}
		if (egen::is_commutative(
			(egen::_GENERATED_OPCODE) func.get_opcode().code_))
		{
			std::sort(hshs.begin(), hshs.end());
		}
		std::unordered_map<std::string,std::string> attrs;
		auto keys = func.ls_attrs();
		for (auto key : keys)
		{
			if (auto value = func.get_attr(key))
			{
				if (auto tref = dynamic_cast<const teq::TensorRef*>(value))
				{
					auto ref = tref->get_tensor();
					ref->accept(*this);
					attrs.emplace(key, boost::uuids::to_string(
						hashes_.at(ref.get())));
				}
				else
				{
					attrs.emplace(key, value->to_string());
				}
			}
		}
		encode_label(&func, func.shape().to_string() + "|" +
			func.to_string() + "\\" +
			fmts::to_string(attrs.begin(), attrs.end()) + "\\" +
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
		[](teq::FuncptrT func)
		{
			auto sess = eigen::get_session();
			sess.track({func});
			sess.update_target({func.get()});
			T* data = (T*) func->data();
			return make_constant(data, func->shape());
		});
}

template <typename T>
void constant_funcs (teq::TensptrsT& roots)
{
	auto sess = eigen::get_session();
	sess.track(roots);
	opt::constant_funcs(roots,
		[&sess](teq::FuncptrT func)
		{
			sess.update_target({func.get()});
			T* data = (T*) func->data();
			return make_constant(data, func->shape());
		});
}

template <typename T>
void optimize (teq::TensptrsT& roots, const opt::CversionCtx& opts)
{
	opt::CustomFilters filters;
	filters.prefilters_.push_back(remove_duplicates<T>);
	filters.prefilters_.push_back(constant_funcs<T>);
	if (opts.dbg_msgs_.size() > 0)
	{
		filters.prenode_filters_.push_back(constant_func<T>);
		filters.postfilters_.push_back(remove_duplicates<T>);
	}
	opt::optimize(roots, opts, filters);
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
