#ifndef MATRIX_H
# define MATRIX_H

# include <iostream>
# include <iomanip>
# include <cstdio>
# include <vector>
# include <algorithm>
# include <stdexcept>
# include <iterator>

# ifdef _OPENMP
#	include <omp.h>
	const uint32_t NUM_THREADS = omp_get_max_threads();
# endif


template<class T=double>
class Matrix {

		std::vector< std::vector <T> >	matrix;
		uint32_t						M;
		uint32_t						N;

		void			resize( uint32_t rows, uint32_t cols, T init=0);

	public:

		// -------------------------------------------------------------------------//
		// Constructors																//
		// -------------------------------------------------------------------------//

		// --> default constructor
		Matrix<T>( uint32_t N=0, uint32_t M=0, T init=0 ): N(N), M(M) {
			resize(N, M, init);
		}


		// --> copy constructor
		Matrix<T>( Matrix<T> const & src ) {
			*this = src;
		}

		// --> 2d vector constructor
		Matrix<T>( std::vector< std::vector<T> > const & vec ) {
			uint32_t cols = vec[0].size();

			for (uint32_t i = 0; i < vec.size(); ++i)
				if (vec[i].size() != cols) throw std::logic_error("Not a Matrix");

			N = vec.size();
			M = cols;
			matrix = vec;
		}

		// --> 2d initializer_list constructor
		Matrix<T>( std::initializer_list< std::initializer_list<T> > const & list ) {
			if (list.size()) {
				N = list.size();
				M = list.begin()->size();
				matrix.resize(N);

				auto it = list.begin();
				for (uint32_t i = 0; i < N; ++i, ++it) {
					if (it->size() != M) throw std::logic_error("Not a Matrix");

					matrix[i].reserve(M);
					matrix[i].insert(matrix[i].end(), it->begin(), it->end());
				}

			} else {
				N = 0;
				M = 0;
			}
		}

		// --> Function constructor
		template<typename Functor>
		Matrix<T>( uint32_t N, uint32_t M, Functor f): N(N), M(M) {
			resize(N, M);

			// This loop is not made parallel because its main use is for rand() 
			// calls, and these are not reliable with openmp
			for (uint32_t i = 0; i < N; ++i) {
				for (uint32_t j = 0; j < M; j++) {
					matrix[i][j] = f();
				}
			}
		}

		// --> "Templated type casting" constructor
		template<class U>
		Matrix<T>( Matrix<U>  const & src ) {
			N = src.getNrows();
			M = src.getMcols();
			resize(N, M);

			#pragma omp parallel for collapse(2) num_threads(NUM_THREADS)
			for (uint32_t i = 0; i < N; ++i) {
				for (uint32_t j = 0; j < M; ++j)
					matrix[i][j] = static_cast<T>(src[i][j]);
			}
		}

		// --> Trivial Destructor
		virtual ~Matrix<T>( void ) {}

		// -------------------------------------------------------------------------//
		// Getters / Setters														//
		// -------------------------------------------------------------------------//

		uint32_t		getNrows( void ) const { return N; }
		uint32_t		getMcols( void ) const { return M; }

		Matrix<T>		getRows( uint32_t i ) const;
		Matrix<T>		getCols( uint32_t j ) const;
		Matrix<T>		getRows( std::vector<int> rows ) const;
		Matrix<T>		getCols( std::vector<int> cols ) const;

		// -------------------------------------------------------------------------//
		// Methods																	//
		// -------------------------------------------------------------------------//

		Matrix<T>&		dropRows( uint32_t i );
		Matrix<T>&		dropCols( uint32_t j );
		Matrix<T>&		dropRows( std::vector<int> rows );
		Matrix<T>&		dropCols( std::vector<int> cols );

		Matrix<T>&		appendVer( T init, uint32_t nrows );
		Matrix<T>&		appendVer( std::vector<T> const & vec );  // throw(logic_error)
		Matrix<T>&		appendVer( Matrix<T> const & other );	  // throw(logic_error)
		Matrix<T>&		appendHor( T init, uint32_t ncols );
		Matrix<T>&		appendHor( std::vector<T> const & vec );  // throw(logic_error)
		Matrix<T>&		appendHor( Matrix<T> const & other );	  // throw(logic_error)

		Matrix<T>		cofact(uint32_t row, uint32_t col) const;
		T				det( void ) const;						  // throw(logic_error)
		Matrix<T>		inv( void ) const;					  // throw(logic_error)
		Matrix<T>		transpose( void ) const;

		static T		det( Matrix<T> const & mat );			  // throw(logic_error)

		// -------------------------------------------------------------------------//
		// Operators																//
		// -------------------------------------------------------------------------//

		std::vector<T>&	operator[]( uint32_t i ) { return this->matrix[i]; }
		std::vector<T>	operator[]( uint32_t i ) const { return this->matrix[i]; }

		Matrix<T>&		operator=( Matrix<T> const & rhs );

		Matrix<T>		operator+( Matrix<T> const & rhs ) const; // throw(logic_error)
		Matrix<T>		operator+( T const & rhs ) const;
		Matrix<T>		operator-( Matrix<T> const & rhs ) const; // throw(logic_error)
		Matrix<T>		operator-( T const & rhs ) const;
		Matrix<T>		operator*( Matrix<T> const & rhs ) const; // throw(logic_error)
		Matrix<T>		operator*( T const & rhs ) const;
		Matrix<T>		operator/( T const & rhs ) const;

		Matrix<T>&		operator+=( Matrix<T> const & rhs );	  // throw(logic_error)
		Matrix<T>&		operator+=( T const & rhs );
		Matrix<T>&		operator-=( Matrix<T> const & rhs );	  // throw(logic_error)
		Matrix<T>&		operator-=( T const & rhs );
		Matrix<T>&		operator*=( Matrix<T> const & rhs );	  // throw(logic_error)
		Matrix<T>&		operator*=( T const & rhs );
		Matrix<T>&		operator/=( T const & rhs );

		bool			operator==( Matrix  const & rhs ) const;

};

template<class T> Matrix<T>			operator*( T const & lhs, Matrix<T> const & rhs );
template<class T> std::ostream &	operator<<( std::ostream & o, Matrix<T> const & rhs );

# include "Matrix.tpp"

#endif
