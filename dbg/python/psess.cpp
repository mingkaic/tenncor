#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "eteq/generated/pyapi.hpp"

// #include "dbg/psess/emit/emitter.hpp"
#include "dbg/psess/stats/inspect.hpp"

namespace py = pybind11;

PYBIND11_MODULE(psess, m)
{
	m.doc() = "dbg teq graphs using interactive grpc session";

	py::class_<dbg::iPlugin> plugin(m, "Plugin");

	auto isess = (py::class_<teq::iSession>)
		py::module::import("eteq.eteq").attr("iSession");

	py::class_<dbg::PluginSession> session(m, "PluginSess", isess);

	session
		.def(py::init([]{ return dbg::PluginSession(eigen::default_device()); }))
		.def("add_plugin",
			[](dbg::PluginSession& self, dbg::iPlugin& plugin)
			{
				self.plugins_.push_back(plugin);
			});

	// py::class_<emit::Emitter,std::shared_ptr<emit::Emitter>>
	// emitter(m, "Emitter", plugin);

	// emitter
	// 	.def(py::init([](std::string host, size_t request_duration, size_t stream_duration)
	// 		{
	// 			return std::make_shared<emit::Emitter>(host,
	// 				emit::ClientConfig{
	// 					std::chrono::milliseconds(request_duration),
	// 					std::chrono::milliseconds(stream_duration),
	// 				});
	// 		}),
	// 		py::arg("host") = "localhost:50051",
	// 		py::arg("request_dur") = 1000,
	// 		py::arg("stream_dur") = 30000)
	// 	.def("join",
	// 		[](emit::Emitter& self)
	// 		{
	// 			self.join();
	// 		},
	// 		"Wait until session finishes sends all requests")
	// 	.def("stop",
	// 		[](emit::Emitter& self)
	// 		{
	// 			self.stop();
	// 		},
	// 		"Inform session requests to stop their tasks "
	// 		"(requests will attempt to wrap up call before terminating)");

	py::class_<stats::Inspector,std::shared_ptr<stats::Inspector>>
	inspector(m, "Inspector", plugin);

	inspector
		.def(py::init())
		.def("add",
			[](stats::Inspector& self, eteq::ETensor<PybindT> inspect, std::string label)
			{
				if (auto f = dynamic_cast<teq::iFunctor*>(inspect.get()))
				{
					self.insps_.emplace(f, label);
				}
			}, py::arg("target"), py::arg("label") = "");
}
