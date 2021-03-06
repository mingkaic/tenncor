
#include "tenncor/python/query_ext.hpp"

void query_ext(py::module& m)
{
	py::class_<pytenncor::Statement> stmt(m, "Statement");
	stmt
		.def(py::init(
		[](eteq::ETensorsT roots)
		{
			teq::TensptrsT rtens(roots.begin(), roots.end());
			return pytenncor::Statement(rtens);
		}),
		"Create query statement from roots")
		.def("find",
		[](pytenncor::Statement& self, const std::string& condition,
			const std::string& sym_cap)
		{
			std::stringstream ss;
			ss << condition;
			query::Node cond;
			query::json_parse(cond, ss);

			auto results = self.sindex_.match(cond);
			teq::RefMapT owners = teq::track_ownrefs(self.tracked_);
			eteq::ETensorsT eresults;
			eresults.reserve(results.size());
			if (sym_cap.empty())
			{
				std::transform(results.begin(), results.end(),
					std::back_inserter(eresults),
					[&](query::QueryResult& result)
					{
						return eteq::ETensor(
							owners.at(result.root_).lock(),
							global::context());
					});
			}
			else
			{
				for (query::QueryResult& result : results)
				{
					teq::iTensor* res;
					if (estd::get(res, result.symbs_, sym_cap))
					{
						eresults.push_back(eteq::ETensor(
							owners.at(res).lock(),
							global::context()));
					}
				}
			}
			return eresults;
		}, py::arg("condition"), py::arg("sym_cap") = "");
}
