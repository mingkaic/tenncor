#include <atomic>

#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>

#include "eteq/session.hpp"

#include "pll/partition.hpp"

#ifndef CCE_ASESS_HPP
#define CCE_ASESS_HPP

namespace pll
{

using SessReqsT = std::vector<std::pair<teq::iOperableFunc*,size_t>>;

using AtomicFulfilMapT = std::unordered_map<
	teq::iOperableFunc*,std::atomic<long>>;

struct Session final : public eteq::iSession
{
	Session (size_t nthreads = 2, OpWeightT weights = OpWeightT()) :
		nthreads_(nthreads), weights_(weights) {}

	std::unordered_set<teq::TensptrT> tracked_;

	void track (teq::TensT roots) override
	{
		tracked_.insert(roots.begin(), roots.end());

		teq::GraphStat stat;
		for (auto& trac : tracked_)
		{
			trac->accept(stat);
		}
		teq::ParentFinder pfinder;
		for (teq::TensptrT& root : roots)
		{
			root->accept(pfinder);
		}

		teq::TensT trackvecs(tracked_.begin(), tracked_.end());
		PartGroupsT groups = k_partition(trackvecs, nthreads_, weights_);
		requirements_.clear();
		for (auto& group : groups)
		{
			SessReqsT reqs;
			reqs.reserve(group.size());
			for (teq::iFunctor* func : group)
			{
				auto& args = func->get_children();
				eteq::TensSetT unique_children;
				for (const teq::FuncArg& arg : args)
				{
					auto tens = arg.get_tensor().get();
					if (0 < stat.graphsize_[tens].upper_) // ignore leaves
					{
						unique_children.emplace(tens);
					}
				}
				reqs.push_back({
					static_cast<teq::iOperableFunc*>(func),
					unique_children.size()
				});
			}
			requirements_.push_back(reqs);
		}

		for (auto& assocs : pfinder.parents_)
		{
			for (auto& parent_pair : assocs.second)
			{
				parents_[assocs.first].emplace(
					static_cast<teq::iOperableFunc*>(parent_pair.first));
			}
		}

		ops_.clear();
		for (auto& tpair : stat.graphsize_)
		{
			if (tpair.second.upper_ > 0)
			{
				ops_.emplace(static_cast<teq::iOperableFunc*>(tpair.first));
			}
		}
	}

	// this function is expected to be called repeatedly during runtime
	void update (eteq::TensSetT updated = {}, eteq::TensSetT ignores = {}) override
	{
		AtomicFulfilMapT fulfilments;
		for (auto op : ops_)
		{
			fulfilments.emplace(op, 0);
		}
		for (teq::iTensor* unodes : updated)
		{
			if (dynamic_cast<teq::iFunctor*>(unodes))
			{
				auto& node_parents = parents_[unodes];
				for (auto& node_parent : node_parents)
				{
					++fulfilments[node_parent];
				}
			}
		}
		// for each req in requirements distribute to thread
		boost::asio::thread_pool pool(nthreads_);
		for (auto& req : requirements_)
		{
			// add thread
			boost::asio::post(pool,
			[this, &req, &fulfilments, &ignores]()
			{
				for (auto& op : req)
				{
					// fulfilled and not ignored
					auto& ff = fulfilments.at(op.first);
					if (ff++ == op.second &&
						false == estd::has(ignores, op.first))
					{
						op.first->update();
						std::unordered_set<teq::iOperableFunc*> op_parents;
						if (estd::get(op_parents,
							this->parents_, op.first))
						{
							for (auto& op_parent : op_parents)
							{
								++fulfilments.at(op_parent);
							}
						}
						++ff;
					}
					--ff;
				}
			});
		}
		pool.join();
	}

	// this function is expected to be called repeatedly during runtime
	void update_target (eteq::TensSetT target, eteq::TensSetT updated = {}) override
	{
		teq::OnceTraveler targetted;
		for (auto& tens : target)
		{
			tens->accept(targetted);
		}
		AtomicFulfilMapT fulfilments;
		for (auto op : ops_)
		{
			fulfilments.emplace(op, 0);
		}
		for (teq::iTensor* unodes : updated)
		{
			if (dynamic_cast<teq::iFunctor*>(unodes))
			{
				auto& node_parents = parents_[unodes];
				for (auto& node_parent : node_parents)
				{
					++fulfilments[node_parent];
				}
			}
		}
		// for each req in requirements distribute to thread
		boost::asio::thread_pool pool(nthreads_);
		for (auto& req : requirements_)
		{
			// make thread
			boost::asio::post(pool,
			[this, &req, &fulfilments, &targetted]()
			{
				for (auto& op : req)
				{
					// is relevant to target, is fulfilled and not ignored
					auto& ff = fulfilments.at(op.first);
					if (ff++ == op.second &&
						estd::has(targetted.visited_, op.first))
					{
						op.first->update();
						std::unordered_set<teq::iOperableFunc*> op_parents;
						if (estd::get(op_parents,
							this->parents_, op.first))
						{
							for (auto& op_parent : op_parents)
							{
								++fulfilments.at(op_parent);
							}
						}
						++ff;
					}
					--ff;
				}
			});
		}
		pool.join();
	}

	void optimize (const opt::OptCtx& rules)
	{
		teq::TensT tracked(tracked_.begin(), tracked_.end());
		opt::optimize(tracked, rules);
		parents_.clear();
		track(tracked);
	}

	std::vector<SessReqsT> requirements_;

	std::unordered_map<teq::iTensor*,
		std::unordered_set<teq::iOperableFunc*>> parents_;

private:
	size_t nthreads_;

	OpWeightT weights_;

	std::unordered_set<teq::iOperableFunc*> ops_;
};

}

#endif // CCE_ASESS_HPP
