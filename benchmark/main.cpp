#include <chrono>
#include <iostream>

#include "ade/coord.hpp"

int main (int argc, char** argv)
{
    ade::CoordptrT reducer = ade::reduce(4, {12, 13, 14, 15});
    ade::Shape shape({255, 255, 255, 255, 255, 255, 255, 255});
    ade::Shape outshape({255, 255, 255, 255, 255/12, 255/13, 255/14, 255/15});
    ade::NElemT n = shape.n_elems();
    ade::CoordT incoord;
    ade::CoordT coord;
    ade::NElemT outidx;

    for (ade::NElemT i = 0; i < n; ++i)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        incoord = ade::coordinate(shape, i);
        auto coord_time = std::chrono::high_resolution_clock::now();
        reducer->forward(coord.begin(), incoord.begin());
        auto mapping_time = std::chrono::high_resolution_clock::now();
        outidx = ade::index(outshape, coord);
        auto index_time = std::chrono::high_resolution_clock::now();
        std::cout
            << std::chrono::duration_cast<std::chrono::nanoseconds>(
                coord_time - start_time).count() << ","
            << std::chrono::duration_cast<std::chrono::nanoseconds>(
                mapping_time - coord_time).count() << ","
            << std::chrono::duration_cast<std::chrono::nanoseconds>(
                index_time - mapping_time).count() << "\n";
    }

    return 0;
}
