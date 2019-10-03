#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"

#include "eteq/generated/pyapi.hpp"

#include "dbg/eteq/custom_functor.hpp"

namespace py = pybind11;

using RawDataT = std::vector<PybindT>;
using RawShapeT = std::vector<py::ssize_t>;
using CustomUnaryF = std::function<RawDataT(const RawDataT&,const RawShapeT&)>;

PYBIND11_MODULE(eteq_mocker, m)
{
	m.doc() = "mock eteq equation graphs";

	m.def("custom_unary",
		[](CustomUnaryF unary, NodeptrT& arg)
		{
			return dbg::make_functor<PybindT>(
				[unary](eteq::TensorT<PybindT>& out, const dbg::DataMapT<PybindT>& args)
				{
					const auto& arg = args.at(0);
					auto output = unary(
						RawDataT(arg.data_, arg.data_ + arg.shape_.n_elems()),
						RawShapeT(arg.shape_.begin(), arg.shape_.end()));
					auto& slist = out.dimensions();
					teq::Shape shape(std::vector<teq::DimT>(
						slist.begin(), slist.end()));
					memcpy(out.data(), output.data(), sizeof(PybindT) * shape.n_elems());
				},
				{eteq::identity_map(arg)});
		});
}
