#include "ead/ead.hpp"

int main (int argc, char** argv)
{
	// in, width, height, batch
    ade::Shape shape({2, 3, 3, 3});
	std::vector<float> data(shape.n_elems());
	std::iota(data.begin(), data.end(), 1);

	// out, in, width, height
    ade::Shape kshape({4, 2, 2, 2});
	std::vector<float> kdata(kshape.n_elems());
	std::iota(kdata.begin(), kdata.end(), 1);

	{
		ead::Session<float> session;
		ead::NodeptrT<float> image = ead::make_constant<float>(data.data(), shape);
		ead::NodeptrT<float> kernel = ead::make_constant<float>(kdata.data(), kshape);

		ade::DimT nfilters = kernel->shape().at(0);
		ead::NodesT<float> convs;
		convs.reserve(nfilters);
		// for (ade::DimT i = 0; i < nfilters; ++i)
		// {
		ade::DimT i = 0;
			auto filter = age::slice(kernel, i, 1, 0);
			session.track(filter);
			session.update();
			float* filter_data = (float*) filter->data();
			auto filter_shape = filter->shape();
			for (size_t i = 0, n = filter_shape.n_elems(); i < n; ++i)
			{
				std::cout << filter_data[i] << "\n";
			}
			std::cout << filter_shape.to_string() << "\n";

			// filter = age::permute(filter, {1, 2, 3, 0});
			// auto conved = filter;
			// convs.push_back(conved);
		// }
	}

	{
		ead::TensorT<float> image = ead::make_tensmap(data.data(), shape);
		ead::TensorT<float> kernel = ead::make_tensmap(kdata.data(), kshape);

		// split
		std::vector<ead::TensMapT<float>> filters;
		{
			auto dims = kernel.dimensions();
			ade::DimT nfilters = dims[0];
			filters.reserve(nfilters);
			ade::Shape filtershape;
			std::copy(dims.begin() + 1, dims.end(), filtershape.begin());
			size_t filtersize = filtershape.n_elems();
			float* data = kernel.data();
			std::array<ade::DimT,8> slice_exts;
			std::copy(filtershape.begin(), filtershape.end(), slice_exts.begin());
			slice_exts[0] = 1;
			for (ade::DimT i = 0; i < nfilters; ++i)
			{
				std::array<ade::DimT,8> offsets{i,0,0,0,0,0,0,0};
				std::cout << "slice " << i << "\n";
				auto sliced = kernel.slice(offsets, slice_exts);
				std::cout << sliced << "\n";

				auto filter = ead::make_tensmap<float>(
					data + i * filtersize, filtershape);
				filters.push_back(filter);
				std::cout << "filter " << i << "\n";
				std::cout << filter << "\n";
			}
		}
		std::vector<ead::TensorT<float>> convs;
		ade::DimT nfilters = filters.size();
		convs.reserve(nfilters);
		ade::ShapeT dims;
		auto it = dims.begin();
		std::fill(it, dims.end(), ade::rank_cap);
		std::iota(it, it + 3, 0);
		for (auto& filter : filters)
		{
			auto conved = image.convolve(filter, dims);
			convs.push_back(conved);
		}

		// join
		std::array<std::pair<int,int>,ade::rank_cap> paddings;
		paddings[0] = std::make_pair(0, nfilters - 1);
		for (uint8_t i = 1; i < ade::rank_cap; ++i)
		{
			paddings[i] = std::make_pair(0, 0);
		}
		ead::TensorT<float> conv = convs[0].pad(paddings);
		for (ade::DimT i = 1; i < nfilters; ++i)
		{
			paddings[0] = std::make_pair(i, nfilters - i - 1);
			conv = conv + convs[i].pad(paddings);
		}

		ead::TensorT<float> image_printable = image;
		ead::TensorT<float> kernel_printable = kernel;
		ead::TensorT<float> conv_printable = conv;
		auto idims = image_printable.dimensions();
		auto kdims = kernel_printable.dimensions();
		auto cdims = conv_printable.dimensions();

		std::cout << "image\n" << image << "\n";
		std::cout << fmts::to_string(idims.begin(), idims.end()) << "\n";
		std::cout << "kernel\n" << kernel << "\n";
		std::cout << fmts::to_string(kdims.begin(), kdims.end()) << "\n";
		std::cout << "convolve\n" << conv << "\n";
		std::cout << fmts::to_string(cdims.begin(), cdims.end()) << "\n";
	}

    return 0;
}
