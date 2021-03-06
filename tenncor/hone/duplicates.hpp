
#ifndef HONE_DUPLICATES_HPP
#define HONE_DUPLICATES_HPP

#include "internal/opt/opt.hpp"

#include "internal/eigen/eigen.hpp"

namespace hone
{

using EqualF = std::function<bool(teq::TensptrT,teq::TensptrT)>;

struct Hasher final : public teq::iOnceTraveler
{
	const std::string& at (teq::iTensor* tens) const
	{
		return estd::must_getf(hashes_, tens,
			"failed to find hash for %s",
			tens->to_string().c_str());
	}

	teq::TensMapT<std::string> hashes_;

private:
	/// Implementation of iOnceTraveler
	void visit_leaf (teq::iLeaf& leaf) override
	{
		if (teq::IMMUTABLE == leaf.get_usage())
		{
			std::string label = leaf.shape().to_string() + "|";
			auto& meta = leaf.get_meta();
			auto data = (const char*) leaf.device().data();
			label += meta.type_label() + std::string(
				data, data + leaf.shape().n_elems() *
				egen::type_size(egen::_GENERATED_DTYPE(meta.type_code())));
			encode_label(&leaf, label);
		}
		else
		{
			hashes_.emplace(&leaf, global::get_generator()->get_str());
		}
	}

	/// Implementation of iOnceTraveler
	void visit_func (teq::iFunctor& func) override
	{
		auto deps = func.get_args();
		types::StringsT hshs;
		hshs.reserve(deps.size());
		teq::multi_visit(*this, deps);
		for (teq::TensptrT dep : deps)
		{
			hshs.push_back(at(dep.get()));
		}
		if (egen::is_commutative(
			(egen::_GENERATED_OPCODE) func.get_opcode().code_))
		{
			std::sort(hshs.begin(), hshs.end());
		}
		types::StrUMapT<std::string> attrs;
		auto keys = func.ls_attrs();
		for (auto key : keys)
		{
			if (auto value = func.get_attr(key))
			{
				if (auto tref = dynamic_cast<const teq::TensorRef*>(value))
				{
					auto ref = tref->get_tensor();
					ref->accept(*this);
					attrs.emplace(key, at(ref.get()));
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
		std::string uuid;
		if (false == estd::get(uuid, uuids_, label))
		{
			uuid = global::get_generator()->get_str();
			uuids_.emplace(label, uuid);
		}
		hashes_.emplace(tens, uuid);
	}

	types::StrUMapT<std::string> uuids_;
};

/// Delete and update equivalent functor and leaves
void merge_dups (opt::GraphInfo& graph, EqualF equals);

void merge_dups (opt::GraphInfo& graph);

}

#endif // HONE_DUPLICATES_HPP
