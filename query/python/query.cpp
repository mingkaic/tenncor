#include <sstream>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "teq/logs.hpp"

#include "eteq/generated/pyapi.hpp"
#include "eteq/etens.hpp"

#include "query/query.hpp"

namespace py = pybind11;

namespace pyquery
{

struct Statement final
{
	std::shared_ptr<query::search::OpTrieT> sindex_;

	teq::TensptrsT tracked_;
};

}

PYBIND11_MODULE(query, m)
{
	LOG_INIT(logs::DefLogger);

	m.doc() = "query teq graphs";

	py::class_<pyquery::Statement> stmt(m, "Statement");
	stmt
		.def(py::init(
			[](eteq::ETensorsT<PybindT> roots)
			{
				teq::TensptrsT rtens(roots.begin(), roots.end());
				pyquery::Statement out{
					std::make_shared<query::search::OpTrieT>(), rtens
				};
				query::search::populate_itable(*(out.sindex_), rtens);
				return out;
			}),
			"Create query statement from roots")
		.def("find",
			[](pyquery::Statement& self, std::string condition)
			{
				teq::TensSetT results;
				std::stringstream ss;
				ss << condition;
				query::Query(*self.sindex_).where(results, ss);
				teq::OwnerMapT owners = teq::track_owners(self.tracked_);
				eteq::ETensorsT<PybindT> eresults;
				eresults.reserve(results.size());
				std::transform(results.begin(), results.end(),
					std::back_inserter(eresults),
					[&](teq::iTensor* result)
					{
						return eteq::ETensor<PybindT>(owners.at(result).lock());
					});
				return eresults;
			});
}
