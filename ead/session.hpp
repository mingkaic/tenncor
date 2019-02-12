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
	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		if (info_.end() == info_.find(leaf))
		{
			info_.emplace(leaf, TensInfo{ParentSetT<T>(), 0});
		}
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (info_.end() == info_.find(func))
		{
			auto efunc = static_cast<Functor<T>*>(func);
			const ade::ArgsT& args = func->get_children();
			std::vector<size_t> arg_heights;
			for (const ade::FuncArg& arg : args)
			{
				ade::iTensor* tens = arg.get_tensor().get();
				tens->accept(*this);
				info_[tens].parents_.emplace(efunc);
				arg_heights.push_back(info_[tens].height_);
			}
			need_update_.emplace(efunc);
			info_.emplace(func, TensInfo{ParentSetT<T>(),
				*std::max_element(
					arg_heights.begin(), arg_heights.end()) + 1});
		}
	}

	void track (NodeptrT<T>& node)
	{
		node->get_tensor()->accept(*this);
	}

	// mark every tensor in updates set as need_update
	// update every functor up until reaching elements in stop_updating set
	void update (std::unordered_set<ade::iTensor*> updates = {})
	{
		for (ade::iTensor* need : updates)
		{
			auto it = info_.find(need);
			if (it != info_.end())
			{
				if (it->second.height_ > 0)
				{
					need_update_.emplace(static_cast<Functor<T>*>(need));
				}
				else
				{
					need_update_.insert(
						it->second.parents_.begin(), it->second.parents_.end());
				}
			}
			else
			{
				logs::warnf("cannot find tensor %s tracked in this session",
					need->to_string().c_str());
			}
		}

		std::vector<Functor<T>*> update_heap(need_update_.begin(), need_update_.end());
		// minimum heights at top of heap
		auto minfunc_comp =
			[this](Functor<T>* lhs, Functor<T>* rhs)
			{
				return this->info_[lhs].height_ > this->info_[rhs].height_;
			};
		std::make_heap(update_heap.begin(), update_heap.end(), minfunc_comp);
		while (update_heap.size() > 0)
		{
			auto f = update_heap.front();
			std::pop_heap(update_heap.begin(), update_heap.end(), minfunc_comp);
			update_heap.pop_back();

			f->update();
			for (Functor<T>* fparent : info_[f].parents_)
			{
				if (need_update_.end() == need_update_.find(fparent))
				{
					update_heap.push_back(fparent);
					std::push_heap(update_heap.begin(), update_heap.end(), minfunc_comp);
					need_update_.emplace(fparent);
				}
			}
		}
		need_update_ = {};
	}

private:
	struct TensInfo
	{
		ParentSetT<T> parents_;
		size_t height_;
	};

	std::unordered_map<ade::iTensor*,TensInfo> info_;

	std::unordered_set<Functor<T>*> need_update_;
};

}

#endif // EAD_SESSION_HPP
