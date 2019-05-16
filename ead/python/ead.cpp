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

namespace py = pybind11;

namespace pyead
{

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
	auto pshape = pyead::c2pshape(tnode->shape());
	return py::array(dtype,
		py::array::ShapeContainer(pshape.begin(), pshape.end()),
		tnode->data());
}

std::vector<PybindT> arr2vec (ade::Shape& outshape, py::array data)
{
	py::buffer_info info = data.request();
	outshape = pyead::p2cshape(info.shape);
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
	py::object node = (py::object)
		py::module::import("ead.age").attr("NodeptrT<PybindT>");

	((py::class_<ead::iNode<PybindT>,ead::NodeptrT<PybindT>>) node)
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
	py::class_<ead::Session<PybindT>> session(m, "Session");
	session
		.def(py::init())
		.def("track", &ead::Session<PybindT>::track, "Track node")
		.def("update",
		[](py::object self, std::vector<ead::NodeptrT<PybindT>> nodes)
		{
			auto sess = self.cast<ead::Session<PybindT>*>();
			std::unordered_set<ade::iTensor*> updates;
			for (ead::NodeptrT<PybindT>& node : nodes)
			{
				updates.emplace(node->get_tensor().get());
			}
			sess->update(updates);
		},
		"Return calculated data",
		py::arg("nodes") = std::vector<ead::NodeptrT<PybindT>>{});

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
	m.def("scalar_constant",
	[](PybindT scalar, std::vector<py::ssize_t> slist)
	{
		return ead::make_constant_scalar<PybindT>(scalar,
			pyead::p2cshape(slist));
	},
	"Return scalar constant node");

	m.def("constant",
	[](py::array data)
	{
		ade::Shape shape;
		std::vector<PybindT> vec = pyead::arr2vec(shape, data);
		return ead::make_constant(vec.data(), shape);
	}, "Return constant node with data");

	m.def("scalar_variable", &ead::make_variable_scalar<PybindT>,
	"Return labelled variable containing numpy data array",
	py::arg("scalar"), py::arg("shape"), py::arg("label") = "");

	m.def("variable",
	[](py::array data, std::string label)
	{
		ade::Shape shape;
		std::vector<PybindT> vec = pyead::arr2vec(shape, data);
		return ead::make_variable(vec.data(), shape, label);
	},
	"Return labelled variable containing numpy data array",
	py::arg("data"), py::arg("label") = "");


	m.def("derive", &ead::derive<PybindT>,
	"Return derivative of first tensor with respect to second tensor (deprecated)");

	m.def("seed",
	[](size_t seed)
	{
		ead::get_engine().seed(seed);
	},
	"Seed internal RNG");
}
