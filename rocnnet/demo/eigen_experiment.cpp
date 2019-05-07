#include "ead/ead.hpp"

int main (int argc, char** argv)
{
	ade::Shape image_shape({3, 3});
	std::vector<float> data(image_shape.n_elems());
	std::iota(data.begin(), data.end(), 1);
	ead::TensorT<float> image = ead::make_tensmap(data.data(), image_shape);

	ade::Shape kernel_shape({3, 2});
	std::vector<float> kdata(kernel_shape.n_elems());
	std::iota(kdata.begin(), kdata.end(), 1);
	ead::TensorT<float> kernel = ead::make_tensmap(kdata.data(), kernel_shape);

	ade::CoordptrT input_shaper(new ade::CoordMap(
		[&kernel_shape](ade::MatrixT fwd)
		{
			for (uint8_t i = 0; i < ade::rank_cap; ++i)
			{
				fwd[i][i] = 1;
			}
			for (uint8_t i = 0; i < ade::rank_cap; ++i)
			{
				fwd[ade::rank_cap][i] = -kernel_shape.at(i) + 1;
			}
		}
	));
	ade::Shape out_shape = ade::apply_shaper(input_shaper, image_shape);
	Eigen::array<ptrdiff_t,ade::rank_cap> dims;
		std::iota(dims.begin(), dims.end(), 0);
	auto conv = image.convolve(kernel, dims);

	std::vector<float> sdata(out_shape.n_elems(), 1);
	ead::TensorT<float> super_composite = ead::make_tensmap(sdata.data(), out_shape);

	std::cout << "image\n" << image << "\n";
	std::cout << "kernel\n" << kernel << "\n";
	std::cout << "convolve\n" << conv << "\n";
	std::cout << "super_composite\n" << super_composite << "\n";

	Eigen::array<ptrdiff_t,ade::rank_cap> patch_dims; // shape of super_composite
	std::copy(out_shape.begin(), out_shape.end(), patch_dims.begin());
	{
		Eigen::array<std::pair<int,int>,ade::rank_cap> paddings;
		for (uint8_t i = 0; i < ade::rank_cap; ++i)
		{
			int paddsize = out_shape.at(i) - 1;
			paddings[i] = std::make_pair(paddsize, paddsize);
		}

		auto padded = kernel.pad(paddings);
		auto patched = padded.extract_patches(patch_dims);
		std::array<bool,ade::rank_cap> revflags;
		std::fill(revflags.begin(), revflags.end(), true);
		std::array<size_t,ade::rank_cap+1> pshape;
		pshape[0] = 1;
		std::copy(out_shape.begin(), out_shape.end(), pshape.begin() + 1);
std::cout << fmts::to_string(pshape.begin(), pshape.end()) << "\n";
		std::array<size_t,ade::rank_cap+1> expansion;
		expansion[0] = image_shape.n_elems();
		std::fill(expansion.begin() + 1, expansion.end(), 1);
		auto partial = super_composite
			.reverse(revflags)
			.reshape(pshape)
			.broadcast(expansion);// * patched;
std::cout << patched << "\n";
std::cout << partial << "\n";
	// 	auto gradwrtimage = partial.sum(std::array<size_t,2>{1, 2})
	// 		.reshape(std::array<size_t,2>{3, 3});

	// 	auto dims = gradwrtimage.dimensions();
	// 	std::cout << "grad:\n" << gradwrtimage << "\n";
	// 	std::cout << fmts::to_string(dims.begin(), dims.end()) << "\n";
	// }

	// {
	// 	auto patched = image.extract_patches(patch_dims);
	// 	auto partial = super_composite
	// 		.reshape(std::array<size_t,3>{4, 1, 1})
	// 		.broadcast(std::array<size_t,3>{1, 2, 2}) * patched;
	// 	auto gradwrtkernel = partial.sum(std::array<size_t,1>{0})
	// 		.reshape(std::array<size_t,2>{2, 2});

	// 	auto dims = gradwrtkernel.dimensions();
	// 	std::cout << "grad:\n" << gradwrtkernel << "\n";
	// 	std::cout << fmts::to_string(dims.begin(), dims.end()) << "\n";
	}

	return 0;
}
