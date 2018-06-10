//
//  matmul.ipp
//  slip
//

#ifdef SLIP_MATMUL_HPP

namespace slip
{

static const uint16_t STRASSEN_THRESHOLD = 256;

static inline size_t min_pad (size_t insize)
{
	size_t counter = 0;
	while (insize> STRASSEN_THRESHOLD)
	{
		insize++;
		insize >>= 1;
		counter ++;
	}
	return insize << counter;
}

template <typename T>
static void cubic_mul (T* c, const T* a, const T* b, size_t dimX, size_t dimY, size_t dimZ, size_t coord_map[4])
{
	for (size_t y = 0; y < dimY; y++)
	{
		for (size_t x = 0; x < dimX; x++)
		{
			c[x+y*dimX] = 0;
			for (size_t z = 0; z < dimZ; z++)
			{
				size_t aidx = coord_map[0] * y + coord_map[1] * z;
				size_t bidx = coord_map[2] * x + coord_map[3] * z;
				c[x + y * dimX] += a[aidx] * b[bidx];
			}
		}
	}
}

// strassen coordinates follow matrix convention (row, col)
template <typename T>
static void strassen (T* c, T* a, T* b, size_t dimPad)
{
	if (dimPad <= STRASSEN_THRESHOLD)
	{
		size_t coord_map[4] = {dimPad, 1, 1, dimPad};
		return cubic_mul(c, a, b, dimPad, dimPad, dimPad, coord_map);
	}
	size_t quadRC = dimPad/2;
	size_t quadSize = quadRC*quadRC;

	// we are given 12 quadrants + 3 for 15 quadrant (14 required + 1 for recursion)
	T* temp = new T[quadSize];
	T* temp2 = new T[quadSize];
	// third buffer for recursive strassen call
	T* temp3 = new T[quadSize];

	// processed during first iteration
	// M3L = A11		(c0)
	// M2L = A21 + A22	(c1)
	// M4L = A22		(c2)
	// M5L = A11 + A12	(c3)
	// M6L = A21 - A11	(temp)
	// M7L = A12 - A22	(temp2)
	// processed during second iteration
	// M2R = B11		(a0)
	// M3R = B12 - B22	(a1) upon third iteration. during second it only B12
	// M4R = B21 - B11	(a2) upon third iteration. during second it only B21
	// M5R = B22		(a3)
	// processed during third iteration
	// M1L = A11 + A22	(b0) = c0 + c2
	// M1R = B11 + B22	(b1) = a0 + a3
	// M6R = B11 + B12	(b2) = a0 + a1 before M3R op
	// M7R = B21 + B22	(b3) = a2 + a3 before M4R op

	// SPACE SAVING METHOD
	// iteration 1: partition a to c such that
	// every element in quadrant (1, 1) ends up in first 1/4th of c, (1, 2) -> second 1/4, etc.
	for (size_t x = 0; x < quadRC; x++)
	{
		for (size_t y = 0; y < quadRC; y++)
		{
			size_t quadidx = x + quadRC * y;
			size_t idx11 = x + dimPad * y;
			size_t idx12 = x + quadRC + dimPad * y;
			size_t idx21 = x + dimPad * (y + quadRC);
			size_t idx22 = x + quadRC + dimPad * (y + quadRC);

			c[quadidx] = a[idx11]; 								// M3L
			c[quadSize + quadidx] = a[idx21] + a[idx22]; 		// M2L
			c[2 * quadSize + quadidx] = a[idx22]; 				// M4L
			c[3 * quadSize + quadidx] = a[idx11] + a[idx12]; 	// M5L
			temp[quadidx] = a[idx21] - a[idx11];				// M6L
			temp2[quadidx] = a[idx12] - a[idx22]; 				// M7L
		}
	}
	// iteration 2: partition b to a (same rule as iteration 1)
	for (size_t x = 0; x < quadRC; x++)
	{
		for (size_t y = 0; y < quadRC; y++)
		{
			size_t quadidx = x + quadRC * y;
			size_t idx11 = x + dimPad * y;
			size_t idx12 = x + quadRC + dimPad * y;
			size_t idx21 = x + dimPad * (y + quadRC);
			size_t idx22 = x + quadRC + dimPad * (y + quadRC);

			a[quadidx] = b[idx11]; 					// M2R
			a[quadSize + quadidx] = b[idx12]; 		// M3R (preliminary)
			a[2 * quadSize + quadidx] = b[idx21]; 	// M4R (preliminary)
			a[3 * quadSize + quadidx] = b[idx22]; 	// M5R
		}
	}
	// iteration 3: finalize calculations
	for (size_t quadidx = 0; quadidx < quadSize; quadidx++)
	{
		b[quadidx] = c[quadidx] + c[2 * quadSize + quadidx]; 								// M1L
		b[quadSize + quadidx] = a[quadidx] + a[3 * quadSize + quadidx]; 					// M1R
		b[2 * quadSize + quadidx] = a[quadidx] + a[quadSize + quadidx]; 					// M6R
		b[3 * quadSize + quadidx] = a[2 * quadSize + quadidx] + a[3 * quadSize + quadidx]; 	// M7R
		a[quadSize + quadidx] -= a[3 * quadSize + quadidx];									// M3R
		a[2 * quadSize + quadidx] -= a[quadidx];											// M4R
	}

	// goal: clear up c for additions in next stage
	// M6 = (A21 - A11) @ (B11 + B12) = M6L @ M6R	(temp3) = (temp) @ (b2)
	// M7 = (A12 - A22) @ (B21 + B22) = M7L @ M7R	(temp) = (temp2) @ (b3)
	// M1 = (A11 + A22) @ (B11 + B22) = M1L @ M1R	(temp2) = (b0) @ (b1)
	// M2 = (A21 + A22) @ B11 = M2L @ M2R			(b0) = (c1) @ (a0)
	// M3 = A11 @ (B12 - B22) = M3L @ M3R			(b1) = (c0) @ (a1)
	// M4 = A22 @ (B21 - B11) = M4L @ M4R			(b2) = (c2) @ (a2)
	// M5 = (A11 + A12) @ B22 = M5L @ M5R			(b3) = (c3) @ (a3)
	strassen(temp3, temp, b + 2 * quadSize, quadRC);
	strassen(temp, temp2, b + 3 * quadSize, quadRC);
	strassen(temp2, b, b + quadSize, quadRC);
	strassen(b, c + quadSize, a, quadRC);
	strassen(b + quadSize, c, a + quadSize, quadRC);
	strassen(b + 2 * quadSize, c + 2 * quadSize, a + 2 * quadSize, quadRC);
	strassen(b + 3 * quadSize, c + 3 * quadSize, a + 3 * quadSize, quadRC);

	// C11 = M1 + M4 - M5 + M7	(temp2) + (b2) - (b3) + (temp)
	// C12 = M3 + M5			(b1) + (b3)
	// C21 = M2 + M4			(b0) + (b2)
	// C22 = M1 - M2 + M3 + M6	(temp2) - (b0) + (b1) + (temp3)
	for (size_t x = 0; x < quadRC; x++)
	{
		for (size_t y = 0; y < quadRC; y++)
		{
			size_t quadidx = x + quadRC * y;
			size_t idx11 = x + dimPad * y; // C11
			size_t idx12 = x + quadRC + dimPad * y; // C12
			size_t idx21 = x + dimPad * (y + quadRC); // C21
			size_t idx22 = x + quadRC + dimPad * (y + quadRC); // C22

			c[idx11] = temp2[quadidx] + b[2 * quadSize + quadidx] - b[3 * quadSize + quadidx] + temp[quadidx];
			c[idx12] = b[quadSize + quadidx] + b[3 * quadSize + quadidx];
			c[idx21] = b[quadidx] + b[2 * quadSize + quadidx];
			c[idx22] = temp2[quadidx] - b[quadidx] + b[quadSize + quadidx] + temp3[quadidx];
		}
	}
	delete [] temp;
	delete [] temp2;
	delete [] temp3;
}


template <typename T>
void matmul (clay::State& dest, std::vector<clay::State> srcs)
{
	assert(2 == srcs.size());
	clay::Shape& ashape = srcs.front().shape_;
	clay::Shape& bshape = srcs.back().shape_;
	T* tdest = safe_get<T>(dest);
	const T* src0 = safe_get<const T>(srcs.front());
	const T* src1 = safe_get<const T>(srcs.back());
	size_t dim_z = ashape.at(0);
	size_t dim_y;
	if (ashape.rank() < 2)
	{
		dim_y = 1;
	}
	else
	{
		dim_y = ashape.at(1);
	}
	size_t dim_x = bshape.at(0);

	// assert that beyond2d is same for A, B, and output C
	size_t beyond2d = srcs[0].shape_.n_elems() / (dim_z * dim_y);

#ifdef ENABLE_STRASSEN // strassen is very cumbersome in a lot of cases
	size_t dim_pad = min_pad(std::max(std::max(dim_x, dim_y), dim_z));
	if (dim_pad> STRASSEN_THRESHOLD)
	{
		for (size_t i = 0; i < beyond2d; i++)
		{
			const T* rawa = src0 + i * (dim_z * dim_y);
			const T* rawb = src1 + i * (dim_x * dim_z);
			T* rawc = tdest + i * (dim_x * dim_y);

			size_t n_mat = dim_pad * dim_pad;
			T* out = new T[n_mat];
			T* a = new T[n_mat];
			T* b = new T[n_mat];
			for (size_t y = 0; y < dim_y; y++)
			{
				for (size_t z = 0; z < dim_z; z++)
				{
					size_t aidx = dim_z * y + z;
					a[z + dim_pad * y] = rawa[aidx];
				}
			}
			for (size_t z = 0; z < dim_z; z++)
			{
				for (size_t x = 0; x < dim_x; x++)
				{
					size_t bidx = x + dim_x * z;
					b[x + dim_pad * z] = rawb[bidx];
				}
			}
			strassen(out, a, b, dim_pad);
			for (size_t y = 0; y < dim_y; y++)
			{
				std::memcpy(rawc + y * dim_x, out + y * dim_pad, sizeof(T) * dim_x);
			}

			delete [] out;
			delete [] a;
			delete [] b;
		}
		return;
	}
#endif /* ENABLE_STRASSEN */
	for (size_t i = 0; i < beyond2d; i++)
	{
		const T* rawa = src0 + i * (dim_z * dim_y);
		const T* rawb = src1 + i * (dim_x * dim_z);
		T* rawc = tdest + i * (dim_x * dim_y);

		size_t coord_map[4] = {dim_z, 1, 1, dim_x};
		cubic_mul(rawc, rawa, rawb, dim_x, dim_y, dim_z, coord_map);
	}
}

template <typename T>
void jacobian (clay::State& dest, std::vector<clay::State> srcs)
{
	clay::Shape& outshape = dest.shape_;
	clay::State& dstate = srcs[2];
	uint64_t* dims = safe_get<uint64_t>(dstate);
	uint64_t matchdim = dims[0];
	uint64_t swapdim = dims[1];

	clay::State& state = srcs[swapdim];
	T* d = safe_get<T>(dest);
	clay::Shape& inshape = state.shape_;
	const T* s = safe_get<const T>(state);
	size_t innerlimit = srcs[matchdim].shape_.as_list()[swapdim];
	size_t ylimit = srcs[0].shape_.as_list()[0];
	size_t xlimit = srcs[1].shape_.as_list()[0];
	size_t n = inshape.n_elems();
	// zero out background
	std::memset(d, 0, outshape.n_elems() * sizeof(T));
	for (size_t i = 0; i < n; ++i)
	{
		std::vector<size_t> coord = coordinate(inshape, i);
		size_t x = coord[0];
		size_t y = coord[1];
		for (size_t j = 0; j < innerlimit; ++j)
		{
			std::vector<size_t> vx = {x, j};
			std::vector<size_t> vy = {y, j};
			vx[swapdim] *= xlimit;
			vy[swapdim] *= ylimit;
			coord[matchdim] = vx[0] + vx[1];
			coord[swapdim] = vy[0] + vy[1];
			d[index(outshape, coord)] = s[i];
		}
	}
}

}

#endif
