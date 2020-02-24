#include <sstream>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "teq/logs.hpp"

#include "eteq/generated/pyapi.hpp"
#include "eteq/etens.hpp"

#include "query/query.hpp"
#include "query/parse.hpp"

namespace py = pybind11;

namespace pyquery
{

struct Statement final
{
	Statement (teq::TensptrsT tens) : tracked_(tens)
	{
		for (auto ten : tens)
		{
			ten->accept(sindex_);
		}
	}

	query::Query sindex_;

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
				return pyquery::Statement(rtens);
			}),
			"Create query statement from roots")
		.def("find",
			[](pyquery::Statement& self, std::string condition)
			{
				std::stringstream ss;
				ss << condition;
				query::Node cond;
				query::json_parse(cond, ss);

				auto results = self.sindex_.match(cond);
				teq::OwnerMapT owners = teq::track_owners(self.tracked_);
				eteq::ETensorsT<PybindT> eresults;
				eresults.reserve(results.size());
				std::transform(results.begin(), results.end(),
					std::back_inserter(eresults),
					[&](query::QueryResult& result)
					{
						return eteq::ETensor<PybindT>(
							owners.at(result.root_).lock());
					});
				return eresults;
			});
}
