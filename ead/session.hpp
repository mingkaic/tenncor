#include <list>

#include "ead/constant.hpp"
#include "ead/functor.hpp"

#ifndef EAD_SESSION_HPP
#define EAD_SESSION_HPP

namespace ead
{

template <typename T>
using ParentSetT = std::unordered_set<Functor<T>*>;

template <typename T>
struct Session final : public ade::iTraveler
{
	struct UpdateSession final : public ade::iTraveler
	{
		UpdateSession (Session<T>* sess) : sess_(sess) {}

		/// Implementation of iTraveler
		void visit (ade::iLeaf* leaf) override {}

		/// Implementation of iTraveler
		void visit (ade::iFunctor* func) override
		{
			if (visited_.end() == visited_.find(func))
			{
				auto& update_set = sess_->need_update_;
				auto& desc_set = sess_->descendants_[func];
				auto has_update =
					[&update_set](ade::iTensor* tens)
					{
						return update_set.end() != update_set.find(tens);
					};
				if (update_set.end() != update_set.find(func) ||
					std::any_of(desc_set.begin(), desc_set.end(), has_update))
				{
					const ade::ArgsT& args = func->get_children();
					for (const ade::FuncArg& arg : args)
					{
						ade::iTensor* tens = arg.get_tensor().get();
						auto& child_desc_set = sess_->descendants_[tens];
						if (update_set.end() != update_set.find(tens) ||
							std::any_of(child_desc_set.begin(),
								child_desc_set.end(), has_update))
						{
							tens->accept(*this);
						}
					}
					static_cast<Functor<T>*>(func)->update();
				}
				visited_.emplace(func);
			}
		}

		std::unordered_set<ade::iTensor*> visited_;

		Session<T>* sess_;
	};

	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		descendants_.emplace(leaf, std::unordered_set<ade::iTensor*>());
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (descendants_.end() == descendants_.find(func))
		{
			const ade::ArgsT& args = func->get_children();
			auto& desc_set = descendants_[func];
			for (const ade::FuncArg& arg : args)
			{
				ade::iTensor* tens = arg.get_tensor().get();
				tens->accept(*this);
				auto it = descendants_.find(tens);
				if (descendants_.end() != it)
				{
					desc_set.insert(it->second.begin(), it->second.end());
				}
				desc_set.emplace(static_cast<ade::iLeaf*>(tens));
			}
			need_update_.emplace(func);
		}
	}

	void track (NodeptrT<T>& node)
	{
		auto tens = node->get_tensor();
		tens->accept(*this);

		auto& node_desc_set = descendants_[tens.get()];
		std::remove_if(roots_.begin(), roots_.end(),
			[&node_desc_set](ade::iTensor* root)
			{
				return node_desc_set.end() != node_desc_set.find(root);
			});

		if (std::all_of(roots_.begin(), roots_.end(),
			[&](ade::iTensor* root)
			{
				auto& desc_set = descendants_[root];
				return desc_set.end() == desc_set.find(tens.get());
			}))
		{
			roots_.push_back(tens.get());
		}
	}

	// mark every tensor in updates set as need_update
	// update every functor up until reaching elements in stop_updating set
	template <typename USESS = UpdateSession, typename std::enable_if<
		std::is_base_of<ade::iTraveler,USESS>::value>::type* = nullptr>
	USESS update (std::unordered_set<ade::iTensor*> updates = {})
	{
		need_update_.insert(updates.begin(), updates.end());
		USESS sess(this);
		for (auto root : roots_)
		{
			root->accept(sess);
		}
		return sess;
	}

	std::list<ade::iTensor*> roots_;

	std::unordered_set<ade::iTensor*> need_update_;

	// maps functor to leaves
	std::unordered_map<ade::iTensor*,
		std::unordered_set<ade::iTensor*>> descendants_;
};

}

#endif // EAD_SESSION_HPP
