
#include <future>

#include "error/error.hpp"

#include "distrib/reference.hpp"

#ifndef DISTRIB_IMANAGER_HPP
#define DISTRIB_IMANAGER_HPP

namespace estd
{

using StrSetT = std::unordered_set<std::string>;

}

namespace distr
{

struct DistEvaluator;

struct iDistManager
{
	virtual ~iDistManager (void) = default;

	virtual void expose_node (teq::TensptrT tens) = 0;

	virtual std::optional<std::string> lookup_id (
		teq::iTensor* tens) const = 0;

	virtual teq::TensptrT lookup_node (
		error::ErrptrT& err,
		const std::string& id,
		bool recursive = true) = 0;

	virtual std::future<void> remote_evaluate (
		const std::string& cluster_id,
		const estd::StrSetT& node_id) = 0;

	virtual teq::TensSetT find_reachable (
		error::ErrptrT& err,
		const teq::TensSetT& srcs,
		const estd::StrSetT& dests) = 0;

	virtual teq::TensMapT<teq::TensptrT> derive (
		teq::GradMapT& grads,
		const teq::TensptrSetT& roots,
		const teq::TensptrSetT& targets) = 0;

	virtual std::string get_id (void) const = 0;

	virtual DRefptrSetT get_remotes (void) const = 0;
};

using iDistMgrptrT = std::shared_ptr<iDistManager>;

struct ManagerOwner final : public eigen::iOwner
{
	ManagerOwner (iDistMgrptrT mgr) : mgr_(mgr) {}

	void* get_raw (void) override
	{
		return mgr_.get();
	}

	iDistMgrptrT mgr_;
};

}

#endif // DISTRIB_IMANAGER_HPP
