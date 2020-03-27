#include "python/query_ext.hpp"

void query_ext(py::module& m)
{
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
