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
struct iSession
{
	virtual ~iSession (void) = default;

	virtual void track (NodeptrT<T>& node) = 0;
};

template <typename T>
struct Session final : public iSession<T>
{
	struct SessionInfo final : public ade::iTraveler
	{
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

		std::unordered_set<ade::iTensor*> need_update_;

		// maps functor to leaves
		std::unordered_map<ade::iTensor*,
			std::unordered_set<ade::iTensor*>> descendants_;
	};

	struct SessionStep final : public ade::iTraveler
	{
		SessionStep (SessionInfo* info) : info_(info) {}

		/// Implementation of iTraveler
		void visit (ade::iLeaf* leaf) override {}

		/// Implementation of iTraveler
		void visit (ade::iFunctor* func) override
		{
			if (visited_.end() == visited_.find(func))
			{
				auto& update_set = info_->need_update_;
				auto& desc_set = info_->descendants_[func];
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
						auto& child_desc_set = info_->descendants_[tens];
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

		SessionInfo* info_;
	};

	void track (NodeptrT<T>& node) override
	{
		auto tens = node->get_tensor();
		tens->accept(sess_info_);

		auto& node_desc_set = sess_info_.descendants_[tens.get()];
		std::remove_if(roots_.begin(), roots_.end(),
			[&node_desc_set](ade::iTensor* root)
			{
				return node_desc_set.end() != node_desc_set.find(root);
			});

		if (std::all_of(roots_.begin(), roots_.end(),
			[&](ade::iTensor* root)
			{
				auto& desc_set = sess_info_.descendants_[root];
				return desc_set.end() == desc_set.find(tens.get());
			}))
		{
			roots_.push_back(tens.get());
		}
	}

	// mark every tensor in updates set as need_update
	// update every functor up until reaching elements in stop_updating set
	SessionStep update (std::unordered_set<ade::iTensor*> updates = {})
	{
		sess_info_.need_update_.insert(updates.begin(), updates.end());
		SessionStep step(&sess_info_);
		for (auto root : roots_)
		{
			root->accept(step);
		}
		return step;
	}

	std::list<ade::iTensor*> roots_;

	SessionInfo sess_info_;
};

}

#endif // EAD_SESSION_HPP
