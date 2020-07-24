#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "teq/logs.hpp"

#include "dbg/peval/emit/emitter.hpp"
#include "dbg/peval/stats/inspect.hpp"

#include "tenncor/tenncor.hpp"

namespace py = pybind11;

PYBIND11_MODULE(peval, m)
{
	LOG_INIT(logs::DefLogger);

	m.doc() = "dbg teq graphs using interactive grpc evaluator";

	// ==== plugin definition ====
	py::class_<dbg::iPlugin> plugin(m, "Plugin");

	// ==== evaluator ====
	auto ieval = (py::class_<teq::iEvaluator>)
		py::module::import("tenncor").attr("iEvaluator");

	py::class_<dbg::PlugableEvaluator> eval(m, "PlugableEvaluator", ieval);

	eval
		.def(py::init([]{ return dbg::PlugableEvaluator(); }))
		.def("add_plugin",
		[](dbg::PlugableEvaluator& self, dbg::iPlugin& plugin)
		{
			self.add_plugin(plugin);
		});

	// ==== plugins ====
	py::class_<emit::Emitter,std::shared_ptr<emit::Emitter>>
	emitter(m, "Emitter", plugin);

	emitter
		.def(py::init(
		[](std::string host, size_t request_duration, size_t stream_duration)
		{
			return std::make_shared<emit::Emitter>(host,
				emit::ClientConfig{
					std::chrono::milliseconds(request_duration),
					std::chrono::milliseconds(stream_duration),
				});
		}),
		py::arg("host") = "localhost:50051",
		py::arg("request_dur") = 1000,
		py::arg("stream_dur") = 30000)
		.def("join",
		[](emit::Emitter& self)
		{
			self.join();
		},
		"Wait until evaluator finishes sends all requests")
		.def("stop",
		[](emit::Emitter& self)
		{
			self.stop();
		},
		"Force evaluator requests to stop their tasks "
		"(requests will attempt to wrap up call before terminating)");

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
		},
		py::arg("target"),
		py::arg("label") = "");
}
