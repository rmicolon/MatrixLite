template<class T>
void			Matrix<T>::resize( uint32_t rows, uint32_t cols, T init ) {
	N = rows;
	M = cols;
	matrix.resize(rows);
	
	#pragma omp parallel for num_threads(NUM_THREADS)
	for (uint32_t i = 0; i < rows; ++i) {
		(init) ? matrix[i].resize(cols, init) : matrix[i].resize(cols);
	}
}

template<class T>
Matrix<T>		Matrix<T>::getRows( uint32_t i ) const {
	if (i >= N)
		throw std::logic_error("Index exceeds Matrix dimensions");

	std::vector<std::vector<T>> vec = { matrix[i] };

	return (Matrix<T>(vec));
}

template<class T>
Matrix<T>		Matrix<T>::getCols( uint32_t j ) const {
	if (j >= M)
		throw std::logic_error("Index exceeds Matrix dimensions");

	auto lambda = [this, j, i=0]() mutable { return matrix[i++][j]; };

	return (Matrix<T>(N, 1, lambda));
}

template<class T>
Matrix<T>		Matrix<T>::getRows( std::vector<int> rows ) const {
	Matrix<T>	ret = this->getRows(rows[0]);

	for (auto it = rows.begin() + 1; it != rows.end(); ++it) {
		ret.appendVer(this->getRows(*it));
	}
	return ret;
}

template<class T>
Matrix<T>		Matrix<T>::getCols( std::vector<int> cols ) const {
	Matrix<T>	ret = this->getCols(cols[0]);

	for (auto it = cols.begin() + 1; it != cols.end(); ++it) {
		ret.appendHor(this->getCols(*it));
	}
	return ret;
}

template<class T>
Matrix<T>&		Matrix<T>::dropRows( uint32_t i ) {
	if (i >= N)
		throw std::logic_error("Index exceeds Matrix dimensions");

	matrix.erase(matrix.begin() + i);
	--N;

	return (*this);
}

template<class T>
Matrix<T>&		Matrix<T>::dropCols( uint32_t j ) {
	if (j >= M)
		throw std::logic_error("Index exceeds Matrix dimensions");

	for (auto it = matrix.begin(); it != matrix.end(); ++it) {
		it->erase((*it).begin() + j);
	}
	--M;

	return (*this);
}

template<class T>
Matrix<T>&		Matrix<T>::dropRows( std::vector<int> rows ) {

	std::sort(rows.begin(), rows.end());

	for (auto it = rows.rbegin(); it != rows.rend(); ++it) {
		this->dropRows(*it);
	}

	return (*this);
}

template<class T>
Matrix<T>&		Matrix<T>::dropCols( std::vector<int> cols ) {

	std::sort(cols.begin(), cols.end());

	for (auto it = cols.rbegin(); it != cols.rend(); ++it) {
		this->dropCols(*it);
	}

	return (*this);
}

template<class T>
Matrix<T> &		Matrix<T>::appendVer( T init, uint32_t nrows ) {
	this->resize(N + nrows, M, init);
	return (*this);
}

template<class T>
Matrix<T> &		Matrix<T>::appendVer( std::vector<T> const & vec ) {
	if (vec.size() != M)
		throw std::logic_error("Vector size is not equal to Matrix cols");

	matrix.resize(N + 1);
	matrix[N] = vec;

	N += 1;

	return (*this);
}

template<class T>
Matrix<T> &		Matrix<T>::appendVer( Matrix<T> const & other ) {
	if (other.getMcols() != M)
		throw std::logic_error("Matrices don't have the same number of cols");

	matrix.resize(N + other.getNrows());

	for (uint32_t i = N, j = 0; i < N + other.getNrows(); ++i, ++j)
		matrix[i] = other[j];

	N += other.getNrows();

	return (*this);
}

template<class T>
Matrix<T> &		Matrix<T>::appendHor( T init, uint32_t ncols ) {

	this->resize(N, M + ncols, init);
	return (*this);
}

template<class T>
Matrix<T> &		Matrix<T>::appendHor( std::vector<T> const & vec ) {
	if (vec.size() != N)
		throw std::logic_error("Vector size is not equal to Matrix rows");

	#pragma omp parallel for num_threads(NUM_THREADS)
	for (uint32_t i = 0; i < N; ++i) {
		matrix[i].resize(M + 1);
		matrix[i][M] = vec[i];
	}

	M += 1;

	return (*this);
}

template<class T>
Matrix<T> &		Matrix<T>::appendHor( Matrix<T> const & other ) {
	if (other.getNrows() != N)
		throw std::logic_error("Matrices don't have the same number of rows");

	#pragma omp parallel for num_threads(NUM_THREADS)
	for (uint32_t i = 0; i < N; ++i) {
		matrix[i].resize(M + other.getMcols());
		for (uint32_t j = 0; j < other.getMcols(); ++j) {
			matrix[i][j+M] = other[i][j];
		}
	}

	M += other.getMcols();

	return (*this);
}

template<class T>
Matrix<T>		Matrix<T>::cofact(uint32_t row, uint32_t col) const {
	Matrix<T>	ret(N-1, M-1);

	for (uint32_t i = 0, k = 0; i < N; ++i) {
		if (i == row) continue;

		for (uint32_t j = 0, l = 0; j < M; ++j) {
			if (j == col) continue;

			ret[k][l] = matrix[i][j];
			++l;
		}
		++k;
	}

	return ret;
}

template<class T>
T				Matrix<T>::det( Matrix<T> const & mat ) {
	if (mat.getNrows() != mat.getMcols())
		throw std::logic_error("Non-square Matrix");
	if (mat.getNrows() == 0)
		throw std::logic_error("Empty Matrix");

	if (mat.getNrows() == 1) return mat[0][0];
	if (mat.getNrows() == 2) return mat[0][0]*mat[1][1] - mat[0][1]*mat[1][0];

	T		ret = 0;
	int8_t	sign;

	for (uint32_t i = 0; i < mat.getNrows(); ++i) {
		sign = i % 2 ? -1 : 1;
		ret += (sign * mat[0][i] * mat.cofact(0, i).det());
	}

	return ret;
}

template<class T>
T				Matrix<T>::det( void ) const {
	return det(*this);
}

template<class T>
Matrix<T>		Matrix<T>::inv( void ) const {
	if (N != M)
		throw std::logic_error("Non-square Matrix");
	if (this->det() == 0)
		throw std::logic_error("Non-invertible matrix (Determinant = 0)");

	Matrix<T>	cofactor(N, M);
	int8_t		sign;

	#pragma omp parallel for collapse(2) num_threads(NUM_THREADS)
	for (uint32_t i = 0; i < N; ++i) {
		for (uint32_t j = 0; j < N; ++j) {
			sign = ((i + j) % 2) ? -1 : 1;
			cofactor[i][j] = sign * this->cofact(i, j).det();
		}
	}

	return (cofactor.transpose() / this->det());
}

template<class T>
Matrix<T>		Matrix<T>::transpose( void ) const {
	Matrix<T>	ret(M, N);

	#pragma omp parallel for collapse(2) num_threads(NUM_THREADS)
	for (uint32_t i = 0; i < N; ++i) {
		for (uint32_t j = 0; j < M; ++j) {
			ret[j][i] = matrix[i][j];
		}
	}

	return ret;
}

template<class T>
Matrix<T> &		Matrix<T>::operator=( Matrix<T> const & rhs ) {
	N = rhs.getNrows();
	M = rhs.getMcols();
	matrix.resize(N);

	#pragma omp parallel for num_threads(NUM_THREADS)
	for (uint32_t i = 0; i < N; ++i) {
		matrix[i].resize(M);
		matrix[i] = rhs[i];
	}

	return (*this);
}

template<class T>
Matrix<T>		Matrix<T>::operator+( Matrix<T> const & rhs ) const {
	if (rhs.getNrows() != N || rhs.getMcols() != M)
		throw std::logic_error("Matrices don't have the same dimensions");

	Matrix<T>	ret(N, M);

	#pragma omp parallel for collapse(2) num_threads(NUM_THREADS)
	for (uint32_t i = 0; i < N; ++i) {
		for (uint32_t j = 0; j < M; ++j) {
			ret[i][j] = matrix[i][j] + rhs[i][j];
		}
	}

	return ret;
}

template<class T>
Matrix<T>		Matrix<T>::operator+( T const & rhs ) const {
	Matrix<T>	ret(N, M);

	#pragma omp parallel for collapse(2) num_threads(NUM_THREADS)
	for (uint32_t i = 0; i < N; ++i) {
		for (uint32_t j = 0; j < M; ++j) {
			ret[i][j] = matrix[i][j] + rhs;
		}
	}

	return ret;
}

template<class T>
Matrix<T>		Matrix<T>::operator-( Matrix<T> const & rhs ) const {
	if (rhs.getNrows() != N || rhs.getMcols() != M)
		throw std::logic_error("Matrices don't have the same dimensions");

	Matrix<T>	ret(N, M);

	#pragma omp parallel for collapse(2) num_threads(NUM_THREADS)
	for (uint32_t i = 0; i < N; ++i) {
		for (uint32_t j = 0; j < M; ++j) {
			ret[i][j] = matrix[i][j] - rhs[i][j];
		}
	}

	return ret;
}

template<class T>
Matrix<T>		Matrix<T>::operator-( T const & rhs ) const {
	Matrix<T>	ret(N, M);

	#pragma omp parallel for collapse(2) num_threads(NUM_THREADS)
	for (uint32_t i = 0; i < N; ++i) {
		for (uint32_t j = 0; j < M; ++j) {
			ret[i][j] = matrix[i][j] - rhs;
		}
	}

	return ret;
}

template<class T>
Matrix<T>		Matrix<T>::operator*( Matrix<T> const & rhs ) const {
	if (M != rhs.getNrows())
		throw std::logic_error("Matrix A cols != matrix B rows");

	Matrix<T>	ret(N, rhs.getMcols());

	#pragma omp parallel for collapse(3) num_threads(NUM_THREADS)
	for (uint32_t i = 0; i < ret.getNrows(); ++i) {
		for (uint32_t j = 0; j < ret.getMcols(); ++j) {
			for (uint32_t k = 0; k < M; ++k) {
				ret[i][j] += matrix[i][k] * rhs[k][j];
			}
		}
	}

	return ret;
}

template<class T>
Matrix<T>		Matrix<T>::operator*( T const & rhs ) const {
	Matrix<T>	ret(N, M);

	#pragma omp parallel for collapse(2) num_threads(NUM_THREADS)
	for (uint32_t i = 0; i < N; ++i) {
		for (uint32_t j = 0; j < M; ++j) {
			ret[i][j] = matrix[i][j] * rhs;
		}
	}

	return ret;
}

template<class T>
Matrix<T>		Matrix<T>::operator/( T const & rhs ) const {
	if (rhs == 0) throw std::logic_error("Division by 0");

	Matrix<T>	ret(N, M);

	#pragma omp parallel for collapse(2) num_threads(NUM_THREADS)
	for (uint32_t i = 0; i < N; ++i) {
		for (uint32_t j = 0; j < M; ++j) {
			ret[i][j] = matrix[i][j] / rhs;
		}
	}

	return ret;
}

template<class T>
Matrix<T> &		Matrix<T>::operator+=( Matrix<T> const & rhs ) {
	*this = *this + rhs;
	return *this;
}

template<class T>
Matrix<T> &		Matrix<T>::operator+=( T const & rhs ) {
	*this = *this + rhs;
	return *this;
}

template<class T>
Matrix<T> &		Matrix<T>::operator-=( Matrix<T> const & rhs ) {
	*this = *this - rhs;
	return *this;
}

template<class T>
Matrix<T> &		Matrix<T>::operator-=( T const & rhs ) {
	*this = *this - rhs;
	return *this;
}

template<class T>
Matrix<T> &		Matrix<T>::operator*=( Matrix<T> const & rhs ) {
	*this = *this * rhs;
	return *this;
}

template<class T>
Matrix<T> &		Matrix<T>::operator*=( T const & rhs ) {
	*this = *this * rhs;
	return *this;
}

template<class T>
Matrix<T> &		Matrix<T>::operator/=( T const & rhs ) {
	*this = *this / rhs;
	return *this;
}

template<class T>
bool			Matrix<T>::operator==( Matrix<T> const & rhs ) const {
	if (N != rhs.getNrows()) return false;

	for (uint32_t i = 0; i < N; ++i)
		if (matrix[i] != rhs[i]) return false;

	return true;
}

template<class T>
Matrix<T>		operator*( T const & lhs, Matrix<T> const & rhs ) {
	return rhs * lhs;
}

template<class T>
std::ostream &	operator<<( std::ostream & o, Matrix<T> const & rhs ) {
	for (uint32_t i = 0; i < rhs.getNrows(); ++i) {
		o << "[";
			for (uint32_t j = 0; j < rhs.getMcols(); ++j)
				o << " " << rhs[i][j];
		o << " ]" << std::endl;
	}

	return o;
}
