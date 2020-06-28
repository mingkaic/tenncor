///
/// session.hpp
/// ccur
///
/// Purpose:
/// Implement session that runs functor updates concurrently
///

#include <atomic>

#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>

#include "teq/isession.hpp"

#include "ccur/partition.hpp"

#ifndef CCUR_SESS_HPP
#define CCUR_SESS_HPP

namespace ccur
{

/// Vector of operable functors and number of unique non-leaf children
/// Functors are ordered by dependency,
/// such that parents of any node always appears after the node in this vector
using SessReqsT = std::vector<std::pair<teq::iFunctor*,long>>;

/// Same as SessReqsT except as a list
using LSessReqsT = std::list<std::pair<teq::iFunctor*,long>>;

/// Map operable functors to the number of children updated in
/// any update/update_target call
using AtomicFulfilMapT = std::unordered_map<
	teq::iFunctor*,std::atomic<long>>;

/// Session that updates operable functors concurrently
/// across specified a number of jobs
struct Session final : public teq::iSession
{
	Session (size_t nthreads = 2, OpWeightT weights = OpWeightT()) :
		nthreads_(nthreads), weights_(weights) {}

	/// Implementation of iSession
	void track (const teq::TensptrSetT& roots) override
	{
		tracked_.insert(roots.begin(), roots.end());

		teq::GraphStat stat;
		for (auto& trac : tracked_)
		{
			trac->accept(stat);
		}
		teq::ParentFinder pfinder;
		for (teq::iTensor* root : roots)
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
				auto args = func->get_children();
				teq::TensSetT unique_children;
				for (teq::TensptrT tens : args)
				{
					if (0 < stat.graphsize_[tens.get()].upper_) // ignore leaves
					{
						unique_children.emplace(tens.get());
					}
				}
				reqs.push_back({func, unique_children.size()});
			}
			requirements_.push_back(reqs);
		}

		for (auto& assocs : pfinder.parents_)
		{
			for (auto& parent_pair : assocs.second)
			{
				parents_[assocs.first].emplace(
					static_cast<teq::iFunctor*>(parent_pair.first));
			}
		}

		ops_.clear();
		for (auto& tpair : stat.graphsize_)
		{
			if (tpair.second.upper_ > 0)
			{
				ops_.emplace(static_cast<teq::iFunctor*>(tpair.first));
			}
		}
	}

	/// Implementation of iSession
	void update (teq::iDevice& device,
		const teq::TensSetT& ignored = {}) override
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
					auto children = op->get_children();
					for (teq::TensptrT child : children)
					{
						acceptable.emplace(child.get());
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
			std::unordered_set<teq::iFunctor*> op_parents;
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
			[this, &reqs, &fulfilments]
			{
				for (auto& op : reqs)
				{
					// fulfilled and not ignored
					auto& ff = fulfilments.at(op.first);
					if (ff++ == op.second)
					{
						op.first->calc();
						std::unordered_set<teq::iFunctor*> op_parents;
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

	/// Implementation of iSession
	void update_target (teq::iDevice& device,
		const teq::TensSetT& target,
		const teq::TensSetT& ignored = {}) override
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
					auto children = op->get_children();
					for (teq::TensptrT child : children)
					{
						acceptable.emplace(child.get());
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
			std::unordered_set<teq::iFunctor*> op_parents;
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
			[this, &reqs, &fulfilments]
			{
				for (auto& op : reqs)
				{
					// is relevant to target, is fulfilled and not ignored
					auto& ff = fulfilments.at(op.first);
					if (ff++ == op.second)
					{
						op.first->calc();
						std::unordered_set<teq::iFunctor*> op_parents;
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

	/// Implementation of iSession
	void clear (void) override
	{
		ops_.clear();
		tracked_.clear();
		parents_.clear();
		requirements_.clear();
	}

	/// Set of all tensors input through tracked function
	/// The set of roots of all session graphs is a possible subset
	teq::TensptrSetT tracked_;

	/// Map of tensor to the set of the tensor's parents
	teq::TensMapT<std::unordered_set<teq::iFunctor*>> parents_;

	/// Vector of vectors of operable functors specific to each job
	/// See SessReqsT
	std::vector<SessReqsT> requirements_;

private:
	size_t nthreads_;

	OpWeightT weights_;

	std::unordered_set<teq::iFunctor*> ops_;
};

}

#endif // CCUR_SESS_HPP
