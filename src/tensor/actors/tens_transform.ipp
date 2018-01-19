//
//  tens_transform.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2018-01-19.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#ifdef TENNCOR_TENS_TRANSFORM_HPP

namespace nnet
{

template <typename T>
tens_l2norm<T>::tens_l2norm (out_wrapper<void> dest, 
    std::vector<in_wrapper<void> > srcs) :
tens_general<T>(dest, srcs, 
[](out_wrapper<T> dest, std::vector<in_wrapper<T> > srcs)
{
    assert(1 == srcs.size());
    // l2norm = sqrt(sum_i=0:n(sqr(xi)))
    dest.first[0] = std::sqrt(std::accumulate(srcs[0].first, 
        srcs[0].first + dest.second.n_elems(), 0,
        [](T left, T right) { return left + right * right; }));
}) {}

template <typename T>
tens_transpose<T>::tens_transpose (out_wrapper<void> dest, 
    std::vector<in_wrapper<void> > srcs, 
    std::pair<size_t, size_t> axis_swap) :
tens_general<T>(dest, srcs, 
[axis_swap](out_wrapper<T> dest, std::vector<in_wrapper<T> > srcs)
{
    assert(1 == srcs.size());
    size_t n_elems = dest.second.n_elems();
    size_t rank = srcs[0].second.rank();
    if (axis_swap.first >= rank)
    {
        return;
    }
    std::vector<size_t> coords;
    if (axis_swap.second>= rank)
    {
        // we're transposing with a previously non-existent dimension,
        // so there is a 1-1 correspondence between src and dest
        std::memcpy(dest.first, srcs[0].first, n_elems * sizeof(T));
    }
    else
    {
        for (size_t i = 0; i < n_elems; i++)
        {
            coords = dest.second.coordinate_from_idx(i);
            std::swap(coords[axis_swap.first], coords[axis_swap.second]);
            dest.first[i] = srcs[0].first[srcs[0].second.flat_idx(coords)];
        }
    }
}) {}

template <typename T>
tens_fit<T>::tens_fit (out_wrapper<void> dest,
    std::vector<in_wrapper<void> > srcs) :
tens_general<T>(dest, srcs,
[](out_wrapper<T> dest, std::vector<in_wrapper<T> > srcs)
{
    assert(2 == srcs.size());
    std::vector<size_t> src_list = srcs[0].second.as_list();
    size_t minrank = std::min(src_list.size(), srcs[1].second.rank());
    size_t n_elems = dest.second.n_elems();
    std::vector<size_t> coords;
    for (size_t i = 0; i < n_elems; i++)
    {
        coords = dest.second.coordinate_from_idx(i);
        bool inbound = true;
        for (size_t j = 0; inbound && j < minrank; j++)
        {
            inbound = coords[j] < src_list[j];
        }
        for (size_t j = minrank, rank = coords.size(); inbound && j < rank; j++)
        {
            inbound = coords[j] == 0;
        }
        dest.first[i] = inbound ? srcs[0].first[srcs[0].second.flat_idx(coords)] : 0;
    }
}) {}

template <typename T>
tens_extend<T>::tens_extend (out_wrapper<void> dest,
    std::vector<in_wrapper<void> > srcs, 
    size_t index, size_t multiplier) :
tens_general<T>(dest, srcs,
[index, multiplier](out_wrapper<T> dest, std::vector<in_wrapper<T> > srcs)
{
    assert(1 == srcs.size());
    size_t n_elems = dest.second.n_elems();
    size_t dim = srcs[0].second.as_list()[index];
    std::vector<size_t> coords;
    for (size_t i = 0; i < n_elems; i++)
    {
        std::vector<size_t> coords = dest.second.coordinate_from_idx(i);
        coords[index] = coords[index] % dim;

        dest.first[i] = srcs[0].first[srcs[0].second.flat_idx(coords)];
    }
}) {}

template <typename T>
tens_compress<T>::tens_compress (out_wrapper<void> dest,
    std::vector<in_wrapper<void> > srcs, 
    size_t index, BI_TRANS<T> collector) :
tens_general<T>(dest, srcs,
[index, collector](out_wrapper<T> dest, std::vector<in_wrapper<T> > srcs)
{
    assert(1 == srcs.size());
    size_t n_elems = dest.second.n_elems();
    if (srcs[0].second.rank() <= index)
    {
        std::memcpy(dest.first, srcs[0].first, sizeof(T) * n_elems);
    }
    else
    {
        size_t adim = srcs[0].second.as_list()[index];
        std::vector<size_t> coords;
        for (size_t i = 0; i < n_elems; i++)
        {
            coords = dest.second.coordinate_from_idx(i);
            if (index == coords.size())
            {
                coords.push_back(0);
            }
            else if (index == 0)
            {
                std::vector<size_t> temp = coords;
                coords = {0};
                coords.insert(coords.end(), temp.begin(), temp.end());
            }
            else
            {
                coords[index] = 0;
            }
            size_t src_idx = srcs[0].second.flat_idx(coords);
            dest.first[i] = srcs[0].first[src_idx];
            for (size_t j = 1; j < adim; j++)
            {
                coords[index] = j;
                src_idx = srcs[0].second.flat_idx(coords);
                dest.first[i] = collector(dest.first[i], srcs[0].first[src_idx]);
            }
        }
    }
}) {}

template <typename T>
tens_compress<T>::tens_compress (out_wrapper<void> dest,
    std::vector<in_wrapper<void> > srcs, 
    BI_TRANS<T> collector) :
tens_general<T>(dest, srcs,
[collector](out_wrapper<T> dest, std::vector<in_wrapper<T> > srcs)
{
    assert(1 == srcs.size());
    if (size_t n_ins = srcs[0].second.n_elems())
    {
        dest.first[0] = std::accumulate(srcs[0].first + 1, srcs[0].first + n_ins, *srcs[0].first, collector);
    }
}) {}

template <typename T>
tens_argcompress<T>::tens_argcompress (out_wrapper<void> dest,
    std::vector<in_wrapper<void> > srcs, 
    size_t dimension, REDUCE<T> search) :
tens_general<T>(dest, srcs,
[dimension, search](out_wrapper<T> dest, std::vector<in_wrapper<T> > srcs)
{
    assert(1 == srcs.size());
    size_t n_elems = dest.second.n_elems();
    std::vector<size_t> coords;
    for (size_t i = 0; i < n_elems; i++)
    {
        coords = dest.second.coordinate_from_idx(i);
        if (dimension == coords.size())
        {
            coords.push_back(0);
        }
        else if (dimension == 0)
        {
            std::vector<size_t> temp = coords;
            coords = {0};
            coords.insert(coords.end(), temp.begin(), temp.end());
        }
        std::vector<double> search_vec;
        for (size_t j = 0, adim = srcs[0].second.as_list()[dimension]; j < adim; j++)
        {
            coords[dimension] = j;
            search_vec.push_back(srcs[0].first[srcs[0].second.flat_idx(coords)]);
        }
        dest.first[i] = search(search_vec);
    }
}) {}

template <typename T>
tens_argcompress<T>::tens_argcompress (out_wrapper<void> dest,
    std::vector<in_wrapper<void> > srcs, 
    REDUCE<T> search) :
tens_general<T>(dest, srcs,
[search](out_wrapper<T> dest, std::vector<in_wrapper<T> > srcs)
{
    assert(1 == srcs.size());
    size_t n_ins = srcs[0].second.n_elems();
    std::vector<double> search_vec(srcs[0].first, srcs[0].first + n_ins);
    dest.first[0] = search(search_vec);
}) {}

template <typename T>
tens_flip<T>::tens_flip (out_wrapper<void> dest,
    std::vector<in_wrapper<void> > srcs, 
    std::vector<size_t> dims) :
tens_general<T>(dest, srcs,
[dims](out_wrapper<T> dest, std::vector<in_wrapper<T> > srcs)
{
    assert(1 == srcs.size());
    size_t n_elems = dest.second.n_elems();
    std::vector<size_t> outlist = dest.second.as_list();
    for (size_t i = 0; i < n_elems; i++)
    {
        std::vector<size_t> coord = dest.second.coordinate_from_idx(i);
        for (size_t d : dims)
        {
            coord[d] = outlist[d] - coord[d] - 1;
        }
        dest.first[i] = srcs[0].first[srcs[0].second.flat_idx(coord)];
    }
}) {}

template <typename T>
tens_cross_corr2d<T>::tens_cross_corr2d (out_wrapper<void> dest,
    std::vector<in_wrapper<void> > srcs,
    std::pair<size_t, size_t> dims) :
tens_general<T>(dest, srcs,
[dims](out_wrapper<T> dest, std::vector<in_wrapper<T> > srcs)
{
    assert(2 == srcs.size());
    size_t n_elems = dest.second.n_elems();
    std::vector<size_t> outlist = dest.second.as_list();
    std::vector<size_t> inlist = srcs[0].second.as_list();
    size_t firstn = inlist[dims.first] - outlist[dims.first];
    size_t secondn = inlist[dims.second] - outlist[dims.second];
    std::vector<size_t> coord;
    for (size_t i = 0; i < n_elems; i++)
    {
        dest.first[i] = 0;
        coord = dest.second.coordinate_from_idx(i);
        for (size_t j = 0; j < firstn; j++)
        {
            for (size_t k = 0; k < secondn; k++)
            {
                dest.first[i] += srcs[0].first[srcs[0].second.flat_idx(coord)] * srcs[0].first[k * firstn + j];

                coord[dims.second]++;
            }
            coord[dims.first]++;
        }
    }
}) {}

}

#endif
