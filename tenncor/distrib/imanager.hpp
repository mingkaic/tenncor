
#include "distrib/peer_svc.hpp"

#ifndef DISTRIB_IMANAGER_HPP
#define DISTRIB_IMANAGER_HPP

namespace distr
{

struct iDistrManager
{
	virtual ~iDistrManager (void) = default;

	virtual std::string get_id (void) const = 0;

	virtual iPeerService* get_service (const std::string& svc_key) = 0;
};

using iDistrMgrptrT = std::shared_ptr<iDistrManager>;

}

#endif // DISTRIB_IMANAGER_HPP
