
#include "dbg/print/printsvc/service.hpp"
#include "python/distrib_ext.hpp"

#ifdef PYTHON_DISTRIB_EXT_HPP

void distrib_ext (py::module& m)
{
	auto context = (py::class_<estd::ConfigMap<>>) m.attr("Context");
	auto ieval = (py::class_<teq::iEvaluator>) m.attr("iEvaluator");

	// extend the context API
	context
		.def("get_distrmgr",
		[](global::CfgMapptrT& self)
		{
			return tcr::get_distrmgr(self);
		})
		.def("get_distrmgr",
		[](global::CfgMapptrT& self, const distr::iDistrMgrptrT& mgr)
		{
			tcr::set_distrmgr(mgr, self);
			return tcr::get_distrmgr(self);
		});

	// todo: internalize or work in some python-native consul API
	py::class_<ppconsul::Consul,distr::ConsulptrT> consul(m, "Consul");
	consul
		.def(py::init(
		[](const std::string& address)
		{
			return std::make_shared<ppconsul::Consul>(address);
		}),
		py::arg("address") = "0.0.0.0:8500");

	// ==== distrib manager ====
	py::class_<distr::iDistrManager,distr::iDistrMgrptrT> imgr(m, "iDistrManager");
	py::class_<distr::DistrManager,distr::DistrMgrptrT> mgr(m, "DistrManager", imgr);

	imgr
		.def("expose_node",
		[](distr::iDistrManager& self, eteq::ETensor node)
		{
			return distr::get_iosvc(self).expose_node(node);
		})
		.def("lookup_id",
		[](distr::iDistrManager& self, eteq::ETensor node)
		{
			std::string id;
			if (auto found_id = distr::get_iosvc(self).lookup_id(node.get()))
			{
				id = *found_id;
			}
			return id;
		})
		.def("lookup_node",
		[](distr::iDistrManager& self,
			const std::string& id, bool recursive)
		{
			error::ErrptrT err = nullptr;
			teq::TensptrT node = distr::get_iosvc(self).lookup_node(err, id, recursive);
			if (nullptr != err)
			{
				global::errorf("lookup_node err: %s",
					err->to_string().c_str());
			}
			return eteq::ETensor(node);
		},
		py::arg("id"),
		py::arg("recursive") = true)
		.def("alias_node",
		[](distr::iDistrManager& self,
			const std::string& alias, const std::string& id)
		{
			distr::get_iosvc(self).set_alias(alias, id);
		})
		.def("dealias_node",
		[](distr::iDistrManager& self, const std::string& alias)
		{
			distr::get_iosvc(self).id_from_alias(alias);
		})
		.def("derive",
		[](distr::iDistrManager& self,
			eteq::ETensor root,
			const eteq::ETensorsT& targets)
		{
			return tcr::derive_with_manager(self, root, targets);
		})
		.def("get_id",
		[](distr::iDistrManager& self)
		{
			return self.get_id();
		})
		.def("print_ascii",
		[](distr::iDistrManager& self, eteq::ETensor root)
		{
			distr::get_printsvc(self).print_ascii(std::cout, root.get());
		});

	mgr
		.def(py::init(
		[](distr::ConsulptrT consul, size_t port,
			const std::string& alias, const std::string& svc_name,
			global::CfgMapptrT ctx)
		{
			return tcr::ctxualize_distrmgr(consul, port, alias, {
				distr::register_iosvc,
				distr::register_opsvc,
				distr::register_oxsvc,
				distr::register_printsvc,
			}, svc_name, ctx);
		}),
		py::arg("consul"),
		py::arg("port"),
		py::arg("alias") = "",
		py::arg("service_name") = distr::default_service,
		py::arg("ctx") = global::context());

	// ==== evaluator ====
	py::class_<distr::DistrEvaluator> eval(m, "DistrEvaluator", ieval);

	eval
		.def(py::init(
		[](distr::iDistrMgrptrT mgr)
		{
			return distr::DistrEvaluator(*mgr);
		}),
		py::arg("mgr"));

	m
		.def("localize", tcr::localize,
		py::arg("root"), py::arg("stop") = eteq::ETensorsT{},
		py::arg("ctx") = global::context(),
		"Move all remote references under root subgraph to "
		"specified context ignoring all subgraphs in stop")
		.def("set_distrmgr", tcr::set_distrmgr,
		py::arg("mgr"),
		py::arg("ctx") = global::context())
		.def("get_distrmgr", tcr::get_distrmgr,
		py::arg("ctx") = global::context())
		.def("expose_node", tcr::expose_node,
		"Expose tensor across the cluster via distribution manager")
		.def("lookup_id", tcr::lookup_id,
		"Look up the id of a tensor in local distribution manager")
		.def("lookup_node", tcr::lookup_node,
		py::arg("id"), py::arg("ctx") = global::context(),
		"Look up the id of a tensor in local distribution manager");
}

#endif
