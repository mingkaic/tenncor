#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"

#include "ade/ade.hpp"

#include "ead/generated/api.hpp"
#include "ead/generated/pyapi.hpp"
#include "ead/grader.hpp"
#include "ead/constant.hpp"
#include "ead/variable.hpp"
#include "ead/functor.hpp"
#include "ead/session.hpp"
#include "ead/random.hpp"
#include "ead/parse.hpp"

namespace py = pybind11;

namespace pyead
{

// todo: move these to a common file
ade::Shape p2cshape (std::vector<py::ssize_t>& pyshape)
{
	return ade::Shape(std::vector<ade::DimT>(
		pyshape.rbegin(), pyshape.rend()));
}

std::vector<ade::DimT> c2pshape (const ade::Shape& cshape)
{
	auto it = cshape.begin();
	auto et = cshape.end();
	while (it != et && *(et-1) == 1)
	{
		--et;
	}
	std::vector<ade::DimT> fwd(it, et);
	return std::vector<ade::DimT>(fwd.rbegin(), fwd.rend());
}

template <typename T>
py::array typedata_to_array (ead::iNode<PybindT>* tnode, py::dtype dtype)
{
	auto pshape = c2pshape(tnode->shape());
	return py::array(dtype,
		py::array::ShapeContainer(pshape.begin(), pshape.end()),
		tnode->data());
}

std::vector<PybindT> arr2vec (ade::Shape& outshape, py::array data)
{
	py::buffer_info info = data.request();
	outshape = p2cshape(info.shape);
	size_t n = outshape.n_elems();
	auto dtype = data.dtype();
	char kind = dtype.kind();
	py::ssize_t tbytes = dtype.itemsize();
	std::vector<PybindT> vec;
	switch (kind)
	{
		case 'f':
			switch (tbytes)
			{
				case 4: // float32
				{
					float* dptr = static_cast<float*>(info.ptr);
					vec = std::vector<PybindT>(dptr, dptr + n);
				}
				break;
				case 8: // float64
				{
					double* dptr = static_cast<double*>(info.ptr);
					vec = std::vector<PybindT>(dptr, dptr + n);
				}
					break;
				default:
					logs::fatalf("unsupported float type with %d bytes", tbytes);
			}
			break;
		case 'i':
			switch (tbytes)
			{
				case 1: // int8
				{
					int8_t* dptr = static_cast<int8_t*>(info.ptr);
					vec = std::vector<PybindT>(dptr, dptr + n);
				}
					break;
				case 2: // int16
				{
					int16_t* dptr = static_cast<int16_t*>(info.ptr);
					vec = std::vector<PybindT>(dptr, dptr + n);
				}
					break;
				case 4: // int32
				{
					int32_t* dptr = static_cast<int32_t*>(info.ptr);
					vec = std::vector<PybindT>(dptr, dptr + n);
				}
					break;
				case 8: // int64
				{
					int64_t* dptr = static_cast<int64_t*>(info.ptr);
					vec = std::vector<PybindT>(dptr, dptr + n);
				}
					break;
				default:
					logs::fatalf("unsupported integer type with %d bytes", tbytes);
			}
			break;
		default:
			logs::fatalf("unknown dtype %c", kind);
	}
	return vec;
}

}

PYBIND11_MODULE(ead, m)
{
	m.doc() = "ead variables";

	// ==== node ====
	auto node = (py::class_<ead::iNode<PybindT>,ead::NodeptrT<PybindT>>)
		py::module::import("ead.tenncor").attr("NodeptrT<PybindT>");

	node
		.def("__str__",
			[](py::object self)
			{
				auto dnode = self.cast<ead::iNode<PybindT>*>();
				return dnode->get_tensor()->to_string();
			},
			"Return string representation of this tensor instance")
		.def("shape",
			[](py::object self)
			{
				auto dnode = self.cast<ead::iNode<PybindT>*>();
				ade::Shape shape = dnode->get_tensor()->shape();
				auto pshape = pyead::c2pshape(shape);
				std::vector<int> ipshape(pshape.begin(), pshape.end());
				return py::array(ipshape.size(), ipshape.data());
			},
			"Return this instance's shape")
		.def("children",
			[](py::object self)
			{
				auto dnode = self.cast<ead::iNode<PybindT>*>();
				std::vector<ade::TensptrT> tens;
				if (auto f = dynamic_cast<ade::iFunctor*>(
					dnode->get_tensor().get()))
				{
					auto args = f->get_children();
					std::transform(args.begin(), args.end(),
					std::back_inserter(tens),
					[](ade::FuncArg& mten)
					{
						return mten.get_tensor();
					});
				}
				return tens;
			})
		.def("as_tens",
			[](py::object self)
			{
				auto dnode = self.cast<ead::iNode<PybindT>*>();
				return dnode->get_tensor();
			})
		.def("get",
			[](py::object self)
			{
				auto dnode = self.cast<ead::iNode<PybindT>*>();
				return pyead::typedata_to_array<PybindT>(dnode,
					py::dtype::of<PybindT>());
			});

	// ==== session ====
	py::class_<ead::iSession> isess(m, "iSession");
	py::class_<ead::Session> session(m, "Session", isess);

	isess
		.def("track",
			[](py::object self, ead::NodesT<PybindT> roots)
			{
				auto sess = self.cast<ead::iSession*>();
				ade::TensT troots;
				troots.reserve(roots.size());
				std::transform(roots.begin(), roots.end(),
					std::back_inserter(troots),
					[](ead::NodeptrT<PybindT>& node)
					{
						return node->get_tensor();
					});
				sess->track(troots);
			})
		.def("update",
			[](py::object self, std::vector<ead::NodeptrT<PybindT>> nodes)
			{
				auto sess = self.cast<ead::iSession*>();
				std::unordered_set<ade::iTensor*> updates;
				for (ead::NodeptrT<PybindT>& node : nodes)
				{
					updates.emplace(node->get_tensor().get());
				}
				sess->update(updates);
			},
			"Calculate every node in the graph given list of updated data nodes",
			py::arg("nodes") = std::vector<ead::NodeptrT<PybindT>>{})
		.def("update_target",
			[](py::object self, std::vector<ead::NodeptrT<PybindT>> targeted,
				std::vector<ead::NodeptrT<PybindT>> updated)
			{
				auto sess = self.cast<ead::iSession*>();
				std::unordered_set<ade::iTensor*> targets;
				std::unordered_set<ade::iTensor*> updates;
				for (ead::NodeptrT<PybindT>& node : targeted)
				{
					targets.emplace(node->get_tensor().get());
				}
				for (ead::NodeptrT<PybindT>& node : updated)
				{
					updates.emplace(node->get_tensor().get());
				}
				sess->update_target(targets, updates);
			},
			"Calculate node relevant to targets in the graph given list of updated data",
			py::arg("targets"),
			py::arg("updated") = std::vector<ead::NodeptrT<PybindT>>{});

	py::implicitly_convertible<ead::iSession,ead::Session>();
	session
		.def(py::init())
		.def("optimize",
			[](py::object self, std::string filename)
			{
				auto sess = self.cast<ead::Session*>();
				opt::OptCtx rules = ead::parse_file<PybindT>(filename);
				sess->optimize(rules);
			},
			py::arg("filename") = "cfg/optimizations.rules",
			"Optimize using rules for specified filename");

	// ==== constant ====
	py::class_<ead::ConstantNode<PybindT>,std::shared_ptr<ead::ConstantNode<PybindT>>> constant(
		m, "Constant", node);

	py::implicitly_convertible<ead::iNode<PybindT>,ead::ConstantNode<PybindT>>();

	// ==== variable ====
	py::class_<ead::VariableNode<PybindT>,ead::VarptrT<PybindT>> variable(
		m, "Variable", node);

	py::implicitly_convertible<ead::iNode<PybindT>,ead::VariableNode<PybindT>>();

	variable
		.def("assign",
			[](py::object self, py::array data)
			{
				auto var = self.cast<ead::VariableNode<PybindT>*>();
				ade::Shape shape;
				std::vector<PybindT> vec = pyead::arr2vec(shape, data);
				var->assign(vec.data(), shape);
			},
			"Assign numpy data array to variable");

	// ==== inline functions ====
	m
		// constant creation
		.def("scalar_constant",
			[](PybindT scalar, std::vector<py::ssize_t> slist)
			{
				return ead::make_constant_scalar<PybindT>(scalar,
					pyead::p2cshape(slist));
			},
			"Return scalar constant node")
		.def("constant",
			[](py::array data)
			{
				ade::Shape shape;
				std::vector<PybindT> vec = pyead::arr2vec(shape, data);
				return ead::make_constant(vec.data(), shape);
			}, "Return constant node with data")

		// variable creation
		.def("scalar_variable",
			[](PybindT scalar, std::vector<py::ssize_t> slist, std::string label)
			{
				return ead::make_variable_scalar<PybindT>(scalar, pyead::p2cshape(slist), label);
			},
			"Return labelled variable containing numpy data array",
			py::arg("scalar"),
			py::arg("slist"),
			py::arg("label") = "")
		.def("variable",
			[](py::array data, std::string label)
			{
				ade::Shape shape;
				std::vector<PybindT> vec = pyead::arr2vec(shape, data);
				return ead::make_variable(vec.data(), shape, label);
			},
			"Return labelled variable containing numpy data array",
			py::arg("data"),
			py::arg("label") = "")

		// other stuff
		.def("derive", &ead::derive<PybindT>,
			"Return derivative of first tensor with respect to second tensor (deprecated)")
		.def("seed",
			[](size_t seed)
			{
				ead::get_engine().seed(seed);
			},
			"Seed internal RNG");
}
