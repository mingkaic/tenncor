
#include "dbg/distr_ext/print/service.hpp"
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
	py::class_<ppconsul::Consul,pytenncor::ConsulT> consul(m, "Consul");
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
		[](distr::iDistrManager& self, pytenncor::ETensT node)
		{
			return distr::get_iosvc(self).expose_node(node);
		})
		.def("lookup_id",
		[](distr::iDistrManager& self, pytenncor::ETensT node)
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
			return pytenncor::ETensT(node);
		},
		py::arg("id"),
		py::arg("recursive") = true)
		.def("derive",
		[](distr::iDistrManager& self,
			eteq::ETensor<PybindT> root,
			const eteq::ETensorsT<PybindT>& targets)
		{
			return tcr::derive_with_manager(self, root, targets);
		})
		.def("get_id",
		[](distr::iDistrManager& self)
		{
			return self.get_id();
		})
		.def("print_ascii",
		[](distr::iDistrManager& self, eteq::ETensor<PybindT> root)
		{
			distr::get_printsvc(self).print_ascii(std::cout, root.get());
		});

	mgr
		.def(py::init(
		[](pytenncor::ConsulT consul, size_t port,
			std::string service_name, std::string alias)
		{
			auto consulsvc = distr::make_consul(*consul, port, service_name, alias);
			distr::PeerServiceConfig cfg(consulsvc, egrpc::ClientConfig(
					std::chrono::milliseconds(5000),
					std::chrono::milliseconds(10000),
					5
				));
			estd::ConfigMap<> svcs;
			auto iosvc = new distr::DistrIOService(cfg);
			svcs.add_entry<distr::DistrIOService>(distr::iosvc_key,
				[&](){ return iosvc; });
			svcs.add_entry<distr::DistrOpService>(distr::opsvc_key,
				[&](){ return new distr::DistrOpService(cfg, iosvc); });
			svcs.add_entry<distr::DistrPrintService>(distr::printsvc_key,
				[&](){ return new distr::DistrPrintService(cfg, iosvc); });
			return std::make_shared<distr::DistrManager>(distr::ConsulSvcptrT(consulsvc), svcs);
		}),
		py::arg("consul"),
		py::arg("port"),
		py::arg("service_name") = distr::default_service,
		py::arg("alias") = "");

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
		.def("expose_node", tcr::expose_node<PybindT>,
		"Expose tensor across the cluster in distributed evaluator");
}

#endif
