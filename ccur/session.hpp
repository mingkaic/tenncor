#include <atomic>

#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>

#include "eteq/session.hpp"

#include "ccur/partition.hpp"

#ifndef CCE_ASESS_HPP
#define CCE_ASESS_HPP

namespace ccur
{

using LSessReqsT = std::list<std::pair<teq::iOperableFunc*,size_t>>;

using SessReqsT = std::vector<std::pair<teq::iOperableFunc*,size_t>>;

using AtomicFulfilMapT = std::unordered_map<
	teq::iOperableFunc*,std::atomic<long>>;

struct Session final : public eteq::iSession
{
	Session (size_t nthreads = 2, OpWeightT weights = OpWeightT()) :
		nthreads_(nthreads), weights_(weights) {}

	void track (teq::TensptrsT roots) override
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

		teq::TensptrsT trackvecs(tracked_.begin(), tracked_.end());
		PartGroupsT groups = k_partition(trackvecs, nthreads_, weights_);
		requirements_.clear();
		for (auto& group : groups)
		{
			SessReqsT reqs;
			reqs.reserve(group.size());
			for (teq::iFunctor* func : group)
			{
				auto& args = func->get_children();
				teq::TensSetT unique_children;
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
	void update (teq::TensSetT ignored = {}) override
	{
		size_t nthreads = requirements_.size();
		std::vector<LSessReqsT> indep_requirements(nthreads);
		for (size_t i = 0; i < nthreads; ++i)
		{
			auto& reqs = requirements_[i];
			auto& indep_reqs = indep_requirements[i];
			teq::TensSetT acceptable;
			for (auto& root : tracked_)
			{
				acceptable.emplace(root.get());
			}
			// ignored tensors will never populate reqs
			for (auto rit = reqs.rbegin(), ret = reqs.rend();
				rit != ret; ++rit)
			{
				auto& op = rit->first;
				if (estd::has(acceptable, op) &&
					false == estd::has(ignored, op))
				{
					indep_reqs.push_front({op, rit->second});
					auto& children = op->get_children();
					for (auto& child : children)
					{
						acceptable.emplace(child.get_tensor().get());
					}
				}
			}
		}

		AtomicFulfilMapT fulfilments;
		for (auto op : ops_)
		{
			fulfilments.emplace(op, 0);
		}

		for (auto ig : ignored)
		{
			std::unordered_set<teq::iOperableFunc*> op_parents;
			if (estd::get(op_parents, parents_, ig))
			{
				for (auto& op_parent : op_parents)
				{
					++fulfilments.at(op_parent);
				}
			}
		}

		// for each req in requirements distribute to thread
		boost::asio::thread_pool pool(nthreads);
		for (auto& reqs : indep_requirements)
		{
			// add thread
			boost::asio::post(pool,
			[this, &reqs, &fulfilments]()
			{
				for (auto& op : reqs)
				{
					// fulfilled and not ignored
					auto& ff = fulfilments.at(op.first);
					if (ff++ == op.second)
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
	void update_target (teq::TensSetT target,
		teq::TensSetT ignored = {}) override
	{
		size_t nthreads = requirements_.size();
		std::vector<LSessReqsT> indep_requirements(nthreads);
		for (size_t i = 0; i < nthreads; ++i)
		{
			auto& reqs = requirements_[i];
			auto& indep_reqs = indep_requirements[i];
			teq::TensSetT acceptable;
			for (auto& root : target)
			{
				acceptable.emplace(root);
			}
			// ignored tensors will never populate reqs
			for (auto rit = reqs.rbegin(), ret = reqs.rend();
				rit != ret; ++rit)
			{
				auto& op = rit->first;
				if (estd::has(acceptable, op) &&
					false == estd::has(ignored, op))
				{
					indep_reqs.push_front({op, rit->second});
					auto& children = op->get_children();
					for (auto& child : children)
					{
						acceptable.emplace(child.get_tensor().get());
					}
				}
			}
		}

		AtomicFulfilMapT fulfilments;
		for (auto op : ops_)
		{
			fulfilments.emplace(op, 0);
		}

		for (auto ig : ignored)
		{
			std::unordered_set<teq::iOperableFunc*> op_parents;
			if (estd::get(op_parents, parents_, ig))
			{
				for (auto& op_parent : op_parents)
				{
					++fulfilments.at(op_parent);
				}
			}
		}

		// for each req in requirements distribute to thread
		boost::asio::thread_pool pool(nthreads);
		for (auto& reqs : indep_requirements)
		{
			// make thread
			boost::asio::post(pool,
			[this, &reqs, &fulfilments]()
			{
				for (auto& op : reqs)
				{
					// is relevant to target, is fulfilled and not ignored
					auto& ff = fulfilments.at(op.first);
					if (ff++ == op.second)
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
		teq::TensptrsT tracked(tracked_.begin(), tracked_.end());
		opt::optimize(tracked, rules);
		parents_.clear();
		track(tracked);
	}

	teq::TensptrSetT tracked_;

	std::unordered_map<teq::iTensor*,
		std::unordered_set<teq::iOperableFunc*>> parents_;

	std::vector<SessReqsT> requirements_;

private:
	size_t nthreads_;

	OpWeightT weights_;

	std::unordered_set<teq::iOperableFunc*> ops_;
};

}

#endif // CCE_ASESS_HPP
