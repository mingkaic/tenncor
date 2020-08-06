
#include "dbg/distr_ext/manager.hpp"

#include "python/distrib_ext.hpp"

#ifdef PYTHON_DISTRIB_EXT_HPP

void distrib_ext (py::module& m)
{
	auto context = (py::class_<eigen::TensContext>) m.attr("Context");
	auto ieval = (py::class_<teq::iEvaluator>) m.attr("iEvaluator");

	// extend the context API
	context
		.def("get_manager",
		[](eigen::CtxptrT& self)
		{
			return tcr::get_distmgr(self);
		})
		.def("set_manager",
		[](eigen::CtxptrT& self, const distr::iDistMgrptrT& mgr)
		{
			tcr::set_distmgr(mgr, self);
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
	py::class_<distr::iDistrManager,distr::iDistMgrptrT> imgr(m, "iDistrManager");
	py::class_<distr::DistrManager,distr::DistMgrptrT> mgr(m, "DistrManager", imgr);

	imgr
		.def("expose_node",
		[](distr::iDistrManager& self, pytenncor::ETensT node)
		{
			return self.get_io().expose_node(node);
		})
		.def("lookup_id",
		[](distr::iDistrManager& self, pytenncor::ETensT node)
		{
			std::string id;
			if (auto found_id = self.get_io().lookup_id(node.get()))
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
			teq::TensptrT node = self.get_io().lookup_node(err, id, recursive);
			if (nullptr != err)
			{
				teq::errorf("lookup_node err: %s",
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
		});

	mgr
		.def(py::init(
		[](pytenncor::ConsulT consul, size_t port,
			std::string service_name, std::string alias)
		{
			return std::make_shared<distr::DistrManager>(
				*consul, port, service_name, alias);
		}),
		py::arg("consul"),
		py::arg("port"),
		py::arg("service_name") = distr::default_service,
		py::arg("alias") = "");


	py::class_<distr::DistrDbgManager,distr::DistDbgMgrptrT> dmgr(m, "DistrDbgManager", imgr);
    dmgr
		.def(py::init(
		[](std::shared_ptr<ppconsul::Consul> consul, size_t port,
			std::string service_name, std::string alias)
		{
			return std::make_shared<distr::DistrDbgManager>(
				*consul, port, service_name, alias);
		}),
		py::arg("consul"),
		py::arg("port"),
		py::arg("service_name") = distr::default_service,
		py::arg("alias") = "")
        .def("print_ascii",
        [](distr::DistrDbgManager& self, eteq::ETensor<PybindT> root)
        {
            self.get_print().print_ascii(std::cout, root.get());
		});

	// ==== evaluator ====
	py::class_<distr::DistrEvaluator> eval(m, "DistrEvaluator", ieval);

	eval
		.def(py::init(
		[](distr::iDistMgrptrT mgr)
		{
			return distr::DistrEvaluator(*mgr);
		}),
		py::arg("mgr"));

	m
		.def("expose_node", tcr::expose_node<PybindT>,
		"Expose tensor across the cluster in distributed evaluator");
}

#endif
