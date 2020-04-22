#include "pybind11/functional.h"

#include "python/eteq_ext.hpp"

#ifdef PYTHON_ETEQ_EXT_HPP

using ETensKeysT = std::unordered_map<std::string,eteq::ETensor<PybindT>>;

teq::Session& get_default_sess (void)
{
	static teq::Session sess = eigen::get_session();
	return sess;
}

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
			[](const pyeteq::ETensT& self)
			{
				return self->to_string();
			})
		.def("__hash__",
			[](const pyeteq::ETensT& self)
			{
				return size_t(self.get());
			})
		.def("shape",
			[](const pyeteq::ETensT& self)
			{
				teq::Shape shape = self->shape();
				auto pshape = pyutils::c2pshape(shape);
				std::vector<int> ipshape(pshape.begin(), pshape.end());
				return py::array(ipshape.size(), ipshape.data());
			},
			"Return this instance's shape")
		.def("get",
			[](const pyeteq::ETensT& self)
			{
				return pyeteq::typedata_to_array<PybindT>(
					*self, py::dtype::of<PybindT>());
			})

		// layer extensions
		.def("get_input", eteq::get_input<PybindT>)
		.def("connect", eteq::connect<PybindT>)
		.def("deep_clone", eteq::deep_clone<PybindT>)
		.def("get_storage",
			[](const pyeteq::ETensT& self)
			{
				auto contents = eteq::get_storage<PybindT>(self);
				return std::vector<eteq::EVariable<PybindT>>(
					contents.begin(), contents.end());
			})

		// useful for debugging
		.def("tag",
			[](pyeteq::ETensT& self,
				const std::string& key, const std::string& val)
			{
				if (auto f = dynamic_cast<teq::iFunctor*>(self.get()))
				{
					f->add_attr(key, std::make_unique<marsh::String>(val));
				}
			});

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
			"Calculate etens relevant to targets in the "
			"graph given list of nodes to ignore",
			py::arg("targeted"),
			py::arg("ignored") = std::vector<pyeteq::ETensT>{})
		.def("get_tracked",
			[](teq::iSession& self)
			{
				auto tracked = self.get_tracked();
				return pyeteq::ETensorsT(tracked.begin(), tracked.end());
			})
		.def("replace",
			[](teq::iSession& self, const std::vector<std::pair<
				pyeteq::ETensT,pyeteq::ETensT>>& converts)
			{
				auto tracked = self.get_tracked();
				opt::GraphInfo graph(teq::TensptrsT(
					tracked.begin(), tracked.end()));
				teq::TensMapT<teq::TensptrT> conversions;
				for (auto& convert : converts)
				{
					conversions.emplace(convert.first.get(), convert.second);
				}
				graph.replace(conversions);
				self.clear();
				self.track(graph.get_roots());
			});

	py::implicitly_convertible<teq::iSession,teq::Session>();
	session
		.def(py::init(&eigen::get_session));

	m.attr("global_default_sess") = &get_default_sess();

	// ==== variable ====
	py::class_<eteq::EVariable<PybindT>,pyeteq::ETensT> evar(m, "EVariable");

	evar
		.def(py::init(
			[](py::list slist, PybindT scalar, const std::string& label)
			{
				return eteq::make_variable_scalar<PybindT>(
					scalar, pyutils::p2cshape(slist), label);
			}),
			py::arg("shape"),
			py::arg("scalar") = 0,
			py::arg("label") = "")
		.def("assign",
			[](eteq::EVariable<PybindT>& self, py::array data, teq::iSession& sess)
			{
				teq::ShapedArr<PybindT> arr;
				pyutils::arr2shapedarr(arr, data);
				self->assign(arr, sess);
			},
			"Assign numpy data array to variable",
			py::arg("data"), py::arg("sess") = get_default_sess());

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
				return eteq::make_variable_scalar<PybindT>(
					scalar, pyutils::p2cshape(slist), label);
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
				return eteq::make_variable(
					arr.data_.data(), arr.shape_, label);
			},
			"Return labelled variable containing numpy data array",
			py::arg("data"),
			py::arg("label") = "")
		.def("to_variable",
			[](const pyeteq::ETensT& tens)
			{
				return eteq::EVariable<PybindT>(std::dynamic_pointer_cast<
					eteq::Variable<PybindT>>((teq::TensptrT) tens));
			})

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

		// ==== use eigen randomizer ====
		.def("unif_gen",
			[](PybindT lower, PybindT upper)
			{
				return py::cpp_function(
					eigen::Randomizer().unif_gen<PybindT>(lower, upper));
			}, py::arg("lower") = 0, py::arg("upper") = 1)
		.def("norm_gen",
			[](PybindT mean, PybindT stdev)
			{
				return py::cpp_function(
					eigen::Randomizer().norm_gen<PybindT>(mean, stdev));
			})

		// ==== serialization ====
		.def("load_from_file",
			[](const std::string& filename,
				const std::unordered_map<std::string,size_t>& key_prec)
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

				std::vector<std::string> precids;
				std::vector<std::string> root_ids;
				precids.reserve(key_prec.size());
				root_ids.reserve(roots.size());
				for (teq::TensptrT root : roots)
				{
					std::string id = ids.left.at(root);
					if (estd::has(key_prec, id))
					{
						precids.push_back(id);
					}
					else
					{
						root_ids.push_back(id);
					}
				}
				std::sort(precids.begin(), precids.end(),
					[&key_prec](const std::string& a, const std::string& b)
					{ return key_prec.at(a) < key_prec.at(b); });

				eteq::ETensorsT<PybindT> out;
				out.reserve(roots.size());
				for (const std::string& id : precids)
				{
					out.push_back(ids.right.at(id));
				}
				for (const std::string& id : root_ids)
				{
					out.push_back(ids.right.at(id));
				}
				return out;
			},
			py::arg("filename"),
			py::arg("key_prec") = std::unordered_map<std::string,size_t>{})
		.def("save_to_file",
			[](const std::string& filename,
				const eteq::ETensorsT<PybindT>& models,
				const ETensKeysT& keys)
			{
				std::ofstream output(filename);
				if (false == output.is_open())
				{
					teq::fatalf("file %s not found", filename.c_str());
				}
				onnx::ModelProto pb_model;
				onnx::TensIdT identified;
				for (auto keyit : keys)
				{
					identified.insert({keyit.second.get(), keyit.first});
				}
				eteq::save_model(pb_model, teq::TensptrsT(
					models.begin(), models.end()), identified);
				return pb_model.SerializeToOstream(&output);
			},
			py::arg("filename"), py::arg("models"),
			py::arg("keys") = ETensKeysT{})
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
				eteq::save_model(pb_model, teq::TensptrsT(
					troots.begin(), troots.end()), ids);
				return pb_model.SerializeToOstream(&output);
			});
}

#endif
