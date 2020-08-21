
#include "internal/opt/opt.hpp"

#include "tenncor/eteq/eteq.hpp"

#ifndef TENNCOR_OPT_DUPLICATES_HPP
#define TENNCOR_OPT_DUPLICATES_HPP

namespace eteq
{

using EqualF = std::function<bool(teq::TensptrT,teq::TensptrT)>;

/// Delete and update equivalent functor and leaves
void merge_dups (opt::GraphInfo& graph, EqualF equals);

template <typename T>
struct Hasher final : public teq::iOnceTraveler
{
	const boost::uuids::uuid& at (teq::iTensor* tens) const
	{
		return estd::must_getf(hashes_, tens,
			"failed to find hash for %s",
			tens->to_string().c_str());
	}

	teq::TensMapT<boost::uuids::uuid> hashes_;

private:
	/// Implementation of iOnceTraveler
	void visit_leaf (teq::iLeaf& leaf) override
	{
		if (teq::IMMUTABLE == leaf.get_usage())
		{
			std::string label = leaf.shape().to_string() + "|";
			T* data = (T*) leaf.device().data();
			label += fmts::to_string(data, data + leaf.shape().n_elems());
			encode_label(&leaf, label);
		}
		else
		{
			hashes_.emplace(&leaf, global::get_uuidengine()());
		}
	}

	/// Implementation of iOnceTraveler
	void visit_func (teq::iFunctor& func) override
	{
		auto deps = func.get_dependencies();
		std::vector<std::string> hshs;
		hshs.reserve(deps.size());
		teq::multi_visit(*this, deps);
		for (teq::TensptrT dep : deps)
		{
			hshs.push_back(boost::uuids::to_string(
				at(dep.get())));
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
						at(ref.get())));
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
			uuid = global::get_uuidengine()();
			uuids_.emplace(label, uuid);
		}
		hashes_.emplace(tens, uuid);
	}

	std::unordered_map<std::string,boost::uuids::uuid> uuids_;
};

template <typename T>
void merge_dups (opt::GraphInfo& graph)
{
	Hasher<T> hasher;
	for (auto& root : graph.roots_)
	{
		root->accept(hasher);
	}
	merge_dups(graph,
		[&](teq::TensptrT a, teq::TensptrT b)
		{
			return hasher.at(a.get()) == hasher.at(b.get());
		});
}

}

#endif // TENNCOR_OPT_DUPLICATES_HPP
