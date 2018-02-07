//
//  d_shaped.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2018-01-19.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#ifdef TENNCOR_D_SHAPED_HPP

namespace nnet
{

template <typename T>
void extend (VARR dest, std::vector<VARR> srcs, ARGS args)
{
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcs.front().second);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape = srcs.front().second;
	size_t index = args.front();
	T* d = dest.first;
	T* s = srcs.front().first;
	size_t dim = srcshape.as_list()[index];
	
	size_t n = destshape.n_elems();
	std::vector<size_t> coords;
	for (size_t i = 0; i < n; ++i)
	{
		coords = destshape.coordinate_from_idx(i);
		coords[index] = coords[index] % dim;
		d[i] = s[srcshape.flat_idx(coords)];
	}
}

template <typename T>
void flip (VARR dest, std::vector<VARR> srcs, ARGS args)
{
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcs.front().second);
	tensorshape& destshape = dest.second;
	T* d = dest.first;
	T* s = srcs.front().first;

	size_t n = destshape.n_elems();
	std::vector<size_t> coords;
	std::vector<size_t> outlist = destshape.as_list();
	for (size_t i = 0; i < n; i++)
	{
		coords = destshape.coordinate_from_idx(i);
		for (size_t d : args)
		{
			coords[d] = outlist[d] - coords[d] - 1;
		}
		d[i] = s[srcs.front().second.flat_idx(coords)];
	}
}

template <typename T>
void crosscorr2d (VARR dest, std::vector<VARR> srcs, ARGS args)
{
	// assert(srcs.size() == 2 && dest.second.compatible_with(srcs.front().second);
	T* d = dest.first;
	T* main_s = srcs.front().first;
	T* wind_s = srcs.back().first;
	size_t dim0 = args.front();
	size_t dim1 = args.back();
	tensorshape& destshape = dest.second;
	tensorshape& srcshape = srcs.front().second;
	// tensorshape& windshape = srcs.back().second;
	std::vector<size_t> outlist = destshape.as_list();
	std::vector<size_t> inlist = srcshape.as_list();
	size_t firstn = inlist[dim0] - outlist[dim0];
	size_t secondn = inlist[dim1] - outlist[dim1];
	// assert(windshape.as_list[dim0] == firstn && windshape.as_list[dim1] == secondn);
	std::vector<size_t> coords;
	size_t n = destshape.n_elems();
	for (size_t i = 0; i < n; i++)
	{
		d[i] = 0;
		coords = destshape.coordinate_from_idx(i);
		for (size_t j = 0; j < firstn; j++)
		{
			for (size_t k = 0; k < secondn; k++)
			{
				d[i] += main_s[srcshape.flat_idx(coords)] * wind_s[k * firstn + j];
				coords[dim1]++;
			}
			coords[dim0]++;
		}
	}
}

}

#endif
