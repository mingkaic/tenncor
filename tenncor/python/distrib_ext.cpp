
#include "python/distrib_ext.hpp"

#ifdef PYTHON_DISTRIB_EXT_HPP

void distrib_ext (py::module& m)
{
	auto ieval = (py::class_<teq::iEvaluator>) m.attr("iEvaluator");

	// todo: internalize or work in some python-native consul API
	py::class_<ppconsul::Consul,pytenncor::ConsulT> consul(m, "Consul");
	consul
		.def(py::init(
		[](const std::string& address)
		{
			return std::make_shared<ppconsul::Consul>(address);
		}),
		py::arg("address") = "0.0.0.0:8500");

	// ==== evaluator ====
	py::class_<distr::iDistrEvaluator,distr::iDEvalptrT> ievaluator(m, "iDistrEvaluator", ieval);
	py::class_<distr::DistrEvaluator,distr::DEvalptrT> eval(m, "DistrEvaluator", ievaluator);

	ievaluator
		.def("expose_node",
		[](distr::iDistrEvaluator& self, pytenncor::ETensT node)
		{
			return self.expose_node(node);
		})
		.def("lookup_id",
		[](distr::iDistrEvaluator& self, pytenncor::ETensT node)
		{
			std::string id;
			if (auto found_id = self.lookup_id(teq::TensptrT(node)))
			{
				id = *found_id;
			}
			return id;
		})
		.def("lookup_node",
		[](distr::iDistrEvaluator& self,
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
		[](distr::iDistrEvaluator& self)
		{
			return self.get_id();
		});

	eval
		.def(py::init(
		[](pytenncor::ConsulT consul, size_t port,
			std::string service_name, std::string alias)
		{
			return tcr::make_distreval(
				*consul, port, service_name, alias);
		}),
		py::arg("consul"),
		py::arg("port"),
		py::arg("service_name") = distr::default_service,
		py::arg("alias") = "");

	m
		.def("expose_node", tcr::expose_node<PybindT>,
		"Expose tensor across the cluster in distributed evaluator");
}

#endif
