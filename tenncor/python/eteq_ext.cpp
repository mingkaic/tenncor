
#include "python/eteq_ext.hpp"

#ifdef PYTHON_ETEQ_EXT_HPP

using ETensKeysT = std::unordered_map<std::string,eteq::ETensor<PybindT>>;

void eteq_ext (py::module& m)
{
	py::class_<estd::ConfigMap<>,global::CfgMapptrT> context(m, "Context");

	context
		.def(py::init<>([]() { return std::make_shared<estd::ConfigMap<>>(); }))
		.def("get_actives",
		[](global::CfgMapptrT& self)
		{
			teq::TensptrSetT actives;
			for (auto& r : eteq::get_reg(self))
			{
				actives.emplace(r.second);
			}
			pytenncor::ETensorsT out;
			out.reserve(actives.size());
			for (auto& tens : actives)
			{
				out.push_back(pytenncor::ETensT(tens, self));
			}
			return out;
		})
		.def("replace",
		[](global::CfgMapptrT& self, const std::vector<std::pair<
			pytenncor::ETensT,pytenncor::ETensT>>& converts)
		{
			teq::TensptrsT actives;
			for (auto& r : eteq::get_reg(self))
			{
				actives.push_back(r.second);
			}
			opt::GraphInfo graph(actives);
			teq::OwnMapT conversions;
			for (auto& convert : converts)
			{
				conversions.emplace(convert.first.get(), convert.second);
			}
			graph.replace(conversions);
		});

	m.attr("global_context") = global::context();

	// ==== data and shape ====
	auto shape = (py::class_<teq::Shape>) m.attr("Shape");

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

	py::class_<teq::ShapedArr<PybindT>> sarr(m, "ShapedArr");

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
	auto etens = (py::class_<pytenncor::ETensT>) m.attr("ETensor");

	etens
		.def(py::init(
		[](teq::TensptrT tens, global::CfgMapptrT& ctx)
		{
			return pytenncor::ETensT(tens, ctx);
		}),
		py::arg("tens"),
		py::arg("ctx") = global::context())
		.def("__str__",
		[](const pytenncor::ETensT& self)
		{
			return self->to_string();
		})
		.def("__hash__",
		[](const pytenncor::ETensT& self)
		{
			return size_t(self.get());
		})
		.def("shape",
		[](const pytenncor::ETensT& self)
		{
			teq::Shape shape = self->shape();
			auto pshape = pyutils::c2pshape(shape);
			std::vector<int> ipshape(pshape.begin(), pshape.end());
			return py::array(ipshape.size(), ipshape.data());
		},
		"Return this instance's shape")
		.def("raw",
		[](pytenncor::ETensT& self)
		{
			PybindT* data = self.data();
			return pytenncor::typedata_to_array<PybindT>(data, self->shape(),
				self->get_meta().type_code(), py::dtype::of<PybindT>());
		})
		.def("get",
		[](pytenncor::ETensT& self, teq::TensSetT ignored, size_t max_version)
		{
			PybindT* data = self.calc(ignored, max_version);
			return pytenncor::typedata_to_array<PybindT>(data, self->shape(),
				self->get_meta().type_code(), py::dtype::of<PybindT>());
		},
		py::arg("ignored") = teq::TensSetT{},
		py::arg("max_version") = std::numeric_limits<size_t>::max())
		.def("get_version",
		[](const pytenncor::ETensT& self)
		{
			return self->get_meta().state_version();
		})

		// layer extensions
		.def("get_input", eteq::get_input<PybindT>)
		.def("connect", eteq::connect<PybindT>)
		.def("deep_clone", eteq::deep_clone<PybindT>)
		.def("get_storage",
		[](const pytenncor::ETensT& self)
		{
			auto contents = eteq::get_storage<PybindT>(self);
			eteq::EVariablesT<PybindT> vars;
			vars.reserve(contents.size());
			std::transform(contents.begin(), contents.end(),
				std::back_inserter(vars),
				[](eteq::VarptrT<PybindT> var)
				{
					return eteq::EVariable<PybindT>(var,
						global::context());
				});
			return vars;
		})

		// useful for debugging
		.def("tag",
		[](pytenncor::ETensT& self,
			const std::string& key, const std::string& val)
		{
			if (auto f = dynamic_cast<teq::iFunctor*>(self.get()))
			{
				f->add_attr(key, std::make_unique<marsh::String>(val));
			}
		});

	// ==== evaluator ====
	py::class_<teq::iEvaluator,teq::iEvalptrT> ieval(m, "iEvaluator");
	py::class_<teq::Evaluator,teq::EvalptrT> eval(m, "Evaluator", ieval);

	ieval
		.def("evaluate",
		[](teq::iEvaluator& self, std::vector<pytenncor::ETensT> targeted,
			size_t max_version, std::vector<pytenncor::ETensT> ignored)
		{
			teq::TensSetT targeted_set;
			teq::TensSetT ignored_set;
			for (pytenncor::ETensT& etens : targeted)
			{
				targeted_set.emplace(etens.get());
			}
			for (pytenncor::ETensT& etens : ignored)
			{
				ignored_set.emplace(etens.get());
			}
			eigen::Device device(max_version);
			self.evaluate(device, targeted_set, ignored_set);
		},
		"Calculate etens relevant to targets in the "
		"graph given list of nodes to ignore",
		py::arg("targeted"),
		py::arg("max_version") = std::numeric_limits<size_t>::max(),
		py::arg("ignored") = std::vector<pytenncor::ETensT>{});

	py::implicitly_convertible<teq::iEvaluator,teq::Evaluator>();
	eval
		.def(py::init([]{ return std::make_shared<teq::Evaluator>(); }));

	// ==== variable ====
	py::class_<eteq::EVariable<PybindT>,pytenncor::ETensT> evar(m, "EVariable");

	evar
		.def(py::init(
		[](py::list slist, PybindT scalar,
			const std::string& label, global::CfgMapptrT context)
		{
			return eteq::make_variable_scalar<PybindT>(
				scalar, pyutils::p2cshape(slist), label, context);
		}),
		py::arg("shape"),
		py::arg("scalar") = 0,
		py::arg("label") = "",
		py::arg("ctx") = global::context())
		.def("assign",
		[](eteq::EVariable<PybindT>& self, py::array data,
			global::CfgMapptrT ctx)
		{
			teq::ShapedArr<PybindT> arr;
			pyutils::arr2shapedarr(arr, data);
			self->assign(arr, ctx);
		},
		"Assign numpy data array to variable",
		py::arg("data"),
		py::arg("ctx") = global::context());

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
		[](PybindT scalar, py::list slist,
			const std::string& label, global::CfgMapptrT context)
		{
			return eteq::make_variable_scalar<PybindT>(
				scalar, pyutils::p2cshape(slist), label,
				context);
		},
		"Return labelled variable containing numpy data array",
		py::arg("scalar"),
		py::arg("slist"),
		py::arg("label") = "",
		py::arg("ctx") = global::context())
		.def("variable_like",
		[](PybindT scalar, pytenncor::ETensT like,
			const std::string& label, global::CfgMapptrT context)
		{
			return eteq::make_variable_like<PybindT>(
				scalar, (teq::TensptrT) like, label,
				context);
		},
		"Return labelled variable containing numpy data array",
		py::arg("scalar"),
		py::arg("like"),
		py::arg("label") = "",
		py::arg("ctx") = global::context())
		.def("variable",
		[](py::array data,
			const std::string& label, global::CfgMapptrT context)
		{
			teq::ShapedArr<PybindT> arr;
			pyutils::arr2shapedarr(arr, data);
			return eteq::make_variable(
				arr.data_.data(), arr.shape_, label,
				context);
		},
		"Return labelled variable containing numpy data array",
		py::arg("data"),
		py::arg("label") = "",
		py::arg("ctx") = global::context())
		.def("to_variable",
		[](const pytenncor::ETensT& tens, global::CfgMapptrT context)
		{
			auto ctx = tens.get_context();
			if (nullptr == ctx)
			{
				ctx = context;
			}
			return eteq::EVariable<PybindT>(std::dynamic_pointer_cast<
				eteq::Variable<PybindT>>((teq::TensptrT) tens), ctx);
		},
		py::arg("tens"),
		py::arg("ctx") = global::context())

		// ==== other stuff ====
		.def("derive",
		[](eteq::ETensor<PybindT> root,
			const eteq::ETensorsT<PybindT>& targets)
		{
			return tcr::derive<PybindT>(root, targets);
		},
		"Return derivative of first tensor with respect to second tensor")

		.def("trail",
		[](const pytenncor::ETensT& root,
			const std::vector<pytenncor::ETensPairT>& inps)
		{
			teq::OwnMapT inputs;
			for (const auto& inp : inps)
			{
				inputs.emplace(inp.first.get(), inp.second);
			}
			return eteq::trail<PybindT>(root, inputs);
		})

		// ==== optimization ====
		.def("optimize",
		[](const std::string& filename, global::CfgMapptrT context)
		{
			eteq::optimize<PybindT>(filename, context);
		},
		py::arg("filename") = "cfg/optimizations.json",
		py::arg("ctx") = global::context(),
		"Optimize using rules for specified filename")

		// ==== configmap ====
		.def("set_log_level", &global::set_log_level,
			py::arg("level"), py::arg("ctx") = global::context(),
			"Set log level")
		.def("get_log_level", &global::get_log_level,
			py::arg("ctx") = global::context(),
			"Get current log level")
		.def("seed", &global::seed,
			py::arg("seed"), py::arg("ctx") = global::context(),
			"Seed internal RNG")

		// ==== use eigen randomizer ====
		.def("unif_gen",
		[](PybindT lower, PybindT upper)
		{
			return py::cpp_function(
				global::Randomizer().unif_gen<PybindT>(lower, upper));
		}, py::arg("lower") = 0, py::arg("upper") = 1)
		.def("norm_gen",
		[](PybindT mean, PybindT stdev)
		{
			return py::cpp_function(
				global::Randomizer().norm_gen<PybindT>(mean, stdev));
		})

		// ==== serialization ====
		.def("load_from_file",
		[](const std::string& filename,
			const std::unordered_map<std::string,size_t>& key_prec)
		{
			std::ifstream input(filename);
			if (false == input.is_open())
			{
				global::fatalf("file %s not found", filename.c_str());
			}
			onnx::ModelProto pb_model;
			if (false == pb_model.ParseFromIstream(&input))
			{
				global::fatalf("failed to parse onnx from %s",
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
				out.push_back(pytenncor::ETensT(ids.right.at(id), global::context()));
			}
			for (const std::string& id : root_ids)
			{
				out.push_back(pytenncor::ETensT(ids.right.at(id), global::context()));
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
				global::fatalf("file %s not found", filename.c_str());
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
		.def("load_context_file",
		[](const std::string& filename, global::CfgMapptrT& ctx)
		{
			std::ifstream input(filename);
			if (false == input.is_open())
			{
				global::fatalf("file %s not found", filename.c_str());
			}
			onnx::ModelProto pb_model;
			if (false == pb_model.ParseFromIstream(&input))
			{
				global::fatalf("failed to parse onnx from %s",
					filename.c_str());
			}
			onnx::TensptrIdT ids;
			auto roots = eteq::load_model(ids, pb_model);
			input.close();
			pytenncor::ETensorsT out;
			out.reserve(roots.size());
			std::transform(roots.begin(), roots.end(),
				std::back_inserter(out),
				[&](teq::TensptrT tens)
				{
					return pytenncor::ETensT(tens, ctx);
				});
			return out;
		})
		.def("save_context_file",
		[](const std::string& filename, global::CfgMapptrT ctx)
		{
			teq::TensptrSetT roots;
			for (auto& r : eteq::get_reg(ctx))
			{
				roots.emplace(r.second);
			}
			std::ofstream output(filename);
			if (false == output.is_open())
			{
				global::fatalf("file %s not found", filename.c_str());
			}
			onnx::ModelProto pb_model;
			onnx::TensIdT ids;
			eteq::save_model(pb_model, teq::TensptrsT(
				roots.begin(), roots.end()), ids);
			return pb_model.SerializeToOstream(&output);
		});
}

#endif