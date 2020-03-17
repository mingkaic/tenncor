#include "python/eteq_ext.hpp"

#ifdef PYTHON_ETEQ_EXT_HPP

void eteq_ext (py::module& m)
{
	// ==== data and shape ====
	py::class_<teq::Shape> shape(m, "Shape");
	py::class_<teq::ShapedArr<PybindT>> sarr(m, "ShapedArr");

	py::implicitly_convertible<py::list,teq::Shape>();
	shape
		.def(py::init(
			[](py::list dims)
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
	auto etens = (py::class_<pyeteq::ETensT>) m.attr("ETensor");

	etens
		.def(py::init<teq::TensptrT>())
		.def("__str__",
			[](pyeteq::ETensT& self)
			{
				return self->to_string();
			})
		.def("shape",
			[](pyeteq::ETensT& self)
			{
				teq::Shape shape = self->shape();
				auto pshape = pyutils::c2pshape(shape);
				std::vector<int> ipshape(pshape.begin(), pshape.end());
				return py::array(ipshape.size(), ipshape.data());
			},
			"Return this instance's shape")
		.def("get",
			[](pyeteq::ETensT& self)
			{
				return pyeteq::typedata_to_array<PybindT>(
					*self, py::dtype::of<PybindT>());
			})

		// layer extensions
		.def("get_input",
			[](pyeteq::ETensT& self, const std::string& layername)
			{
				return eteq::get_input(layername, self);
			})
		.def("connect",
			[](pyeteq::ETensT& self, const std::string& layername, pyeteq::ETensT& input)
			{
				return eteq::connect(layername, self, input);
			})
		.def("deep_clone",
			[](pyeteq::ETensT& self, const std::string& layername)
			{
				return eteq::deep_clone(layername, self);
			})
		.def("get_storage",
			[](pyeteq::ETensT& self, const std::string& layername)
			{
				return eteq::get_storage(layername, self);
			});

	// ==== variable ====
	py::class_<eteq::EVariable<PybindT>,pyeteq::ETensT> evar(m, "EVariable");

	evar
		.def(py::init(
			[](py::list slist, PybindT scalar, const std::string& label)
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
					[](pyeteq::ETensT& etens)
					{
						return etens;
					});
				self->track(troots);
			})
		.def("update",
			[](teq::iSession* self,
				std::vector<pyeteq::ETensT> ignored)
			{
				teq::TensSetT ignored_set;
				for (pyeteq::ETensT& etens : ignored)
				{
					ignored_set.emplace(etens.get());
				}
				self->update(ignored_set);
			},
			"Calculate every etens in the graph given list of nodes to ignore",
			py::arg("ignored") = std::vector<pyeteq::ETensT>{})
		.def("update_target",
			[](teq::iSession* self,
				std::vector<pyeteq::ETensT> targeted,
				std::vector<pyeteq::ETensT> ignored)
			{
				teq::TensSetT targeted_set;
				teq::TensSetT ignored_set;
				for (pyeteq::ETensT& etens : targeted)
				{
					targeted_set.emplace(etens.get());
				}
				for (pyeteq::ETensT& etens : ignored)
				{
					ignored_set.emplace(etens.get());
				}
				self->update_target(targeted_set, ignored_set);
			},
			"Calculate etens relevant to targets in the graph given list of nodes to ignore",
			py::arg("targeted"),
			py::arg("ignored") = std::vector<pyeteq::ETensT>{})
		.def("get_tracked",
			[](teq::iSession& self)
			{
				auto tracked = self.get_tracked();
				return pyeteq::ETensorsT(tracked.begin(), tracked.end());
			});

	py::implicitly_convertible<teq::iSession,teq::Session>();
	session
		.def(py::init(&eigen::get_session));

	// ==== inline functions ====
	m
		// ==== constant creation ====
		.def("scalar_constant",
			[](PybindT scalar, py::list slist)
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
			[](PybindT scalar, py::list slist, const std::string& label)
			{
				return eteq::make_variable_scalar<PybindT>(scalar, pyutils::p2cshape(slist), label);
			},
			"Return labelled variable containing numpy data array",
			py::arg("scalar"),
			py::arg("slist"),
			py::arg("label") = "")
		.def("variable_like",
			[](PybindT scalar, pyeteq::ETensT like, const std::string& label)
			{
				return eteq::make_variable_like<PybindT>(
					scalar, (teq::TensptrT) like, label);
			},
			"Return labelled variable containing numpy data array",
			py::arg("scalar"),
			py::arg("like"),
			py::arg("label") = "")
		.def("variable",
			[](py::array data, const std::string& label)
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

		.def("trail", [](const pyeteq::ETensT& root,
			const std::vector<pyeteq::ETensPairT>& inps)
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
			[](teq::iSession& sess, const std::string& filename)
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
		// 			teq::error("missing logger in global config");
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
		// 			teq::error("missing logger in global config");
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
		// 			teq::error("missing random engine in global config");
		// 			engine = &eigen::default_engine();
		// 		}
		// 		engine->seed(seed);
			},
			"Seed internal RNG")

		// ==== serialization ====
		.def("load_from_file",
			[](const std::string& filename)
			{
				std::ifstream input(filename);
				if (false == input.is_open())
				{
					teq::fatalf("file %s not found", filename.c_str());
				}
				onnx::ModelProto pb_model;
				if (false == pb_model.ParseFromIstream(&input))
				{
					teq::fatalf("failed to parse onnx from %s",
						filename.c_str());
				}
				onnx::TensptrIdT ids;
				auto roots = eteq::load_model(ids, pb_model);
				input.close();
				return eteq::ETensorsT<PybindT>(roots.begin(), roots.end());
			})
		.def("save_to_file",
			[](const std::string& filename, const eteq::ETensorsT<PybindT>& models)
			{
				std::ofstream output(filename);
				if (false == output.is_open())
				{
					teq::fatalf("file %s not found", filename.c_str());
				}
				onnx::ModelProto pb_model;
				eteq::save_model(pb_model, teq::TensptrsT(models.begin(), models.end()));
				return pb_model.SerializeToOstream(&output);
			})
		.def("load_session_file",
			[](const std::string& filename, teq::iSession& sess)
			{
				std::ifstream input(filename);
				if (false == input.is_open())
				{
					teq::fatalf("file %s not found", filename.c_str());
				}
				onnx::ModelProto pb_model;
				if (false == pb_model.ParseFromIstream(&input))
				{
					teq::fatalf("failed to parse onnx from %s",
						filename.c_str());
				}
				onnx::TensptrIdT ids;
				auto roots = eteq::load_model(ids, pb_model);
				sess.track(roots);
				input.close();
			})
		.def("save_session_file",
			[](const std::string& filename, const teq::iSession& sess)
			{
				auto troots = sess.get_tracked();
				std::ofstream output(filename);
				if (false == output.is_open())
				{
					teq::fatalf("file %s not found", filename.c_str());
				}
				onnx::ModelProto pb_model;
				onnx::TensIdT ids;
				eteq::save_model(pb_model, teq::TensptrsT(troots.begin(), troots.end()), ids);
				return pb_model.SerializeToOstream(&output);
			});
}

#endif
