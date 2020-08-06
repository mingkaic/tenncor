
#include "teq/teq.hpp"

#ifndef EIGEN_GLOBAL_HPP
#define EIGEN_GLOBAL_HPP

namespace eigen
{

using TensRegistryT = std::unordered_map<void*,teq::TensptrT>;

struct iOwner
{
	virtual ~iOwner (void) = default;

	virtual void* get_raw (void) = 0;
};

using OwnerptrT = std::unique_ptr<iOwner>;

// todo: move this to teq and merge with teq/config
struct TensContext final
{
	teq::iEvalptrT eval_ = std::make_shared<teq::Evaluator>();

	TensRegistryT registry_;

	estd::StrMapT<OwnerptrT> owners_;
};

using CtxptrT = std::shared_ptr<TensContext>;

using CtxrefT = std::weak_ptr<TensContext>;

CtxptrT& global_context (void);

}

#endif // EIGEN_GLOBAL_HPP
