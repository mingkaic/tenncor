
#include "python/distrib_ext.hpp"

#ifdef PYTHON_DISTRIB_EXT_HPP

void distrib_ext (py::module& m)
{
	auto isess = (py::class_<teq::iSession>) m.attr("iSession");

	// todo: internalize or work in some python-native consul API
	py::class_<ppconsul::Consul,pytenncor::ConsulT> consul(m, "Consul");
	consul
		.def(py::init(
		[](const std::string& address)
		{
			return std::make_shared<ppconsul::Consul>(address);
		}),
		py::arg("address") = "0.0.0.0:8500");

	py::class_<distr::iDistribSess,distr::DSessptrT>
	isession(m, "iDistribSess", isess);
	py::class_<distr::DistribSess,std::shared_ptr<distr::DistribSess>>
	session(m, "DistribSess", isession);

	isession
		.def("lookup_id",
		[](distr::iDistribSess& self, pytenncor::ETensT node)
		{
			std::string id;
			if (auto found_id = self.lookup_id(teq::TensptrT(node)))
			{
				id = *found_id;
			}
			return id;
		})
		.def("lookup_node",
		[](distr::iDistribSess& self,
			const std::string& id, bool recursive)
		{
			error::ErrptrT err = nullptr;
			teq::TensptrT node = self.lookup_node(err, id, recursive);
			if (nullptr != err)
			{
				teq::errorf("lookup_node err: %s",
					err->to_string().c_str());
			}
			return pytenncor::ETensT(node);
		},
		py::arg("id"),
		py::arg("recursive") = true)
		.def("get_id",
		[](distr::iDistribSess& self)
		{
			return self.get_id();
		});
	session
		.def(py::init(
		[](pytenncor::ConsulT consul, size_t port,
			std::string service_name, std::string alias)
		{
			return tcr::make_distrsess(
				*consul, port, service_name, alias);
		}),
		py::arg("consul"),
		py::arg("port"),
		py::arg("service_name") = distr::default_service,
		py::arg("alias") = "");
}

#endif
