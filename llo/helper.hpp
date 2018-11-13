#include "ade/ifunctor.hpp"

#include "llo/operator.hpp"
#include "llo/data.hpp"

#ifndef LLO_HELPER_HPP
#define LLO_HELPER_HPP

namespace llo
{

ade::Tensorptr grad_prod (size_t gradidx, age::TensT tens);

ade::Tensorptr grad_min (size_t gradidx, age::TensT tens);

ade::Tensorptr grad_max (size_t gradidx, age::TensT tens);

ade::CoordPtrT reduce (uint8_t rank, const ade::Shape& shape);

using DataArgsT = std::vector<std::pair<ade::CoordPtrT,GenericData>>;

template <typename T>
VecRef<T> to_ref (std::pair<ade::CoordPtrT,GenericData>& gdata)
{
	return VecRef<T>{
		gdata.first,
		(T*) gdata.second.data_.get(),
		gdata.second.shape_
	};
}

template <typename T>
std::vector<VecRef<T>> to_refs (DataArgsT& data)
{
    std::vector<VecRef<T>> args(data.size());
    std::transform(data.begin(), data.end(), args.begin(),
        [](std::pair<ade::CoordPtrT,GenericData>& gd)
        {
            return to_ref<T>(gd);
        });
    return args;
}

}

#endif // LLO_HELPER_HPP
