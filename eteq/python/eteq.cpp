#include "pybind11/stl.h"

#include "teq/logs.hpp"

#include "pyutils/convert.hpp"

#include "eigen/device.hpp"
#include "eigen/random.hpp"

#include "eteq/generated/pyapi.hpp"
#include "eteq/derive.hpp"
#include "eteq/layer.hpp"
#include "eteq/optimize.hpp"

namespace py = pybind11;

namespace pyead
{

using ETensT = eteq::ETensor<PybindT>;

using ETensPairT = std::pair<ETensT,ETensT>;

template <typename T>
py::array typedata_to_array (teq::iTensor& tens, py::dtype dtype)
{
	assert(egen::get_type<PybindT>() == tens.type_code());
	auto pshape = pyutils::c2pshape(tens.shape());
	return py::array(dtype,
		py::array::ShapeContainer(pshape.begin(), pshape.end()),
		tens.device().data());
}

}

PYBIND11_MODULE(eteq, m)
{
	LOG_INIT(logs::DefLogger);
	DEVICE_INIT(eigen::Device);
	RANDOM_INIT;

	m.doc() = "eteq variables";

	// ==== data and shape ====
	py::class_<teq::Shape> shape(m, "Shape");
	py::class_<teq::ShapedArr<PybindT>> sarr(m, "ShapedArr");

	shape
		.def(py::init(
			[](std::vector<py::ssize_t> dims)
			{
				return pyutils::p2cshape(dims);
			}))
		.def("__getitem__",
			[](teq::Shape& shape, size_t idx) { return shape.at(idx); },
			py::is_operator())
		.def("n_elems",
			[](teq::Shape& shape) { return shape.n_elems(); })
		.def("as_list",
			[](teq::Shape& shape) { return pyutils::c2pshape(shape); });

	sarr
		.def(py::init(
			[](py::array ar)
			{
				teq::ShapedArr<PybindT> a;
				pyutils::arr2shapedarr(a, ar);
				return a;
			}))
		.def("as_numpy",
			[](teq::ShapedArr<PybindT>* self) -> py::array
			{
				return pyutils::shapedarr2arr<PybindT>(*self);
			});

	// ==== etens ====
	auto etens = (py::class_<pyead::ETensT>)
		py::module::import("eteq.tenncor").attr("ETensor");

	etens
		.def(py::init<teq::TensptrT>())
		.def("__str__",
			[](pyead::ETensT& self)
			{
				return self->to_string();
			})
		.def("shape",
			[](pyead::ETensT& self)
			{
				teq::Shape shape = self->shape();
				auto pshape = pyutils::c2pshape(shape);
				std::vector<int> ipshape(pshape.begin(), pshape.end());
				return py::array(ipshape.size(), ipshape.data());
			},
			"Return this instance's shape")
		.def("get",
			[](pyead::ETensT& self)
			{
				return pyead::typedata_to_array<PybindT>(
					*self, py::dtype::of<PybindT>());
			});

	// ==== variable ====
	py::class_<eteq::EVariable<PybindT>,pyead::ETensT> evar(m, "EVariable");

	evar
		.def(py::init(
			[](std::vector<py::ssize_t> slist, PybindT scalar, std::string label)
			{
				return eteq::make_variable_scalar<PybindT>(scalar, pyutils::p2cshape(slist), label);
			}),
			py::arg("shape"),
			py::arg("scalar") = 0,
			py::arg("label") = "")
		.def("assign",
			[](eteq::EVariable<PybindT>& self, py::array data)
			{
				teq::ShapedArr<PybindT> arr;
				pyutils::arr2shapedarr(arr, data);
				self->assign(arr);
			},
			"Assign numpy data array to variable");

	// ==== elayer ====
	py::class_<eteq::ELayer<PybindT>> elayer(m, "ELayer");

	elayer
		.def(py::init(
			[](pyead::ETensT root, pyead::ETensT input)
			{
				return eteq::ELayer<PybindT>(
					estd::must_ptr_cast<teq::iFunctor>((teq::TensptrT) root), input);
			}),
			py::arg("root"), py::arg("input"))
		.def("deep_clone", &eteq::ELayer<PybindT>::deep_clone)
		.def("connect", &eteq::ELayer<PybindT>::connect)
		.def("get_storage",
			[](eteq::ELayer<PybindT>& self)
			{
				auto stores = self.get_storage();
				std::vector<eteq::EVariable<PybindT>> estores(stores.begin(), stores.end());
				return estores;
			})
		.def("root",
			[](eteq::ELayer<PybindT>& self) -> pyead::ETensT
			{
				return pyead::ETensT(self.root());
			})
		.def("input", &eteq::ELayer<PybindT>::input);

	// ==== session ====
	py::class_<teq::iSession> isess(m, "iSession");
	py::class_<teq::Session> session(m, "Session", isess);

	isess
		.def("track",
			[](teq::iSession* self, eteq::ETensorsT<PybindT> roots)
			{
				teq::TensptrsT troots;
				troots.reserve(roots.size());
				std::transform(roots.begin(), roots.end(),
					std::back_inserter(troots),
					[](pyead::ETensT& etens)
					{
						return etens;
					});
				self->track(troots);
			})
		.def("update",
			[](teq::iSession* self,
				std::vector<pyead::ETensT> ignored)
			{
				teq::TensSetT ignored_set;
				for (pyead::ETensT& etens : ignored)
				{
					ignored_set.emplace(etens.get());
				}
				self->update(ignored_set);
			},
			"Calculate every etens in the graph given list of nodes to ignore",
			py::arg("ignored") = std::vector<pyead::ETensT>{})
		.def("update_target",
			[](teq::iSession* self,
				std::vector<pyead::ETensT> targeted,
				std::vector<pyead::ETensT> ignored)
			{
				teq::TensSetT targeted_set;
				teq::TensSetT ignored_set;
				for (pyead::ETensT& etens : targeted)
				{
					targeted_set.emplace(etens.get());
				}
				for (pyead::ETensT& etens : ignored)
				{
					ignored_set.emplace(etens.get());
				}
				self->update_target(targeted_set, ignored_set);
			},
			"Calculate etens relevant to targets in the graph given list of nodes to ignore",
			py::arg("targeted"),
			py::arg("ignored") = std::vector<pyead::ETensT>{})
		.def("get_tracked", &teq::iSession::get_tracked);

	py::implicitly_convertible<teq::iSession,teq::Session>();
	session
		.def(py::init(&eigen::get_session));

	// ==== inline functions ====
	m
		// ==== constant creation ====
		.def("scalar_constant",
			[](PybindT scalar, std::vector<py::ssize_t> slist)
			{
				return eteq::make_constant_scalar<PybindT>(scalar,
					pyutils::p2cshape(slist));
			},
			"Return scalar constant etens")
		.def("constant",
			[](py::array data)
			{
				teq::ShapedArr<PybindT> arr;
				pyutils::arr2shapedarr(arr, data);
				return eteq::make_constant(arr.data_.data(), arr.shape_);
			}, "Return constant etens with data")

		// ==== variable creation ====
		.def("scalar_variable",
			[](PybindT scalar, std::vector<py::ssize_t> slist, std::string label)
			{
				return eteq::make_variable_scalar<PybindT>(scalar, pyutils::p2cshape(slist), label);
			},
			"Return labelled variable containing numpy data array",
			py::arg("scalar"),
			py::arg("slist"),
			py::arg("label") = "")
		.def("variable_like",
			[](PybindT scalar, pyead::ETensT like, std::string label)
			{
				return eteq::make_variable_like<PybindT>(
					scalar, (teq::TensptrT) like, label);
			},
			"Return labelled variable containing numpy data array",
			py::arg("scalar"),
			py::arg("like"),
			py::arg("label") = "")
		.def("variable",
			[](py::array data, std::string label)
			{
				teq::ShapedArr<PybindT> arr;
				pyutils::arr2shapedarr(arr, data);
				return eteq::make_variable(arr.data_.data(), arr.shape_, label);
			},
			"Return labelled variable containing numpy data array",
			py::arg("data"),
			py::arg("label") = "")

		// ==== other stuff ====
		.def("derive", &eteq::derive<PybindT>,
			"Return derivative of first tensor with respect to second tensor")

		.def("trail", [](const pyead::ETensT& root,
			const std::vector<pyead::ETensPairT>& inps)
			{
				teq::TensMapT<teq::TensptrT> inputs;
				for (const auto& inp : inps)
				{
					inputs.emplace(inp.first.get(), inp.second);
				}
				return eteq::trail<PybindT>(root, inputs);
			})

		// ==== optimization ====
		.def("optimize",
			[](teq::iSession& sess, std::string filename)
			{
				eteq::optimize<PybindT>(sess, filename);
			},
			py::arg("sess"),
			py::arg("filename") = "cfg/optimizations.json",
			"Optimize using rules for specified filename")

		// // ==== configmap ====
		// .def("set_log_level",
		// 	[](const std::string& level)
		// 	{
		// 		auto logger = static_cast<logs::iLogger*>(
		// 			config::global_config.get_obj(teq::logger_key));
		// 		if (nullptr == logger)
		// 		{
		// 			logs::error("missing logger in global config");
		// 			logs::get_logger().set_log_level(level);
		// 			return;
		// 		}
		// 		if (logger->supports_level(level))
		// 		{
		// 			logger->set_log_level(level);
		// 		}
		// 	})
		// .def("get_log_level",
		// 	[]
		// 	{
		// 		auto logger = static_cast<logs::iLogger*>(
		// 			config::global_config.get_obj(teq::logger_key));
		// 		if (nullptr == logger)
		// 		{
		// 			logs::error("missing logger in global config");
		// 			return logs::get_logger().get_log_level();
		// 		}
		// 		return logger->get_log_level();
		// 	})
		.def("seed",
			[](size_t seed)
			{
		// 		auto engine = static_cast<eigen::EngineT*>(
		// 			config::global_config.get_obj(eigen::rengine_key));
		// 		if (nullptr == engine)
		// 		{
		// 			logs::error("missing random engine in global config");
		// 			engine = &eigen::default_engine();
		// 		}
		// 		engine->seed(seed);
			},
			"Seed internal RNG");
}
