#include "Matrix.h"

int	main( void )
{

	//  ------ CONSTRUCTORS ------
	// The matrix class has various constructors (see Matrix.h)
	// For example you can create a matrix manually via initialization lists...
	Matrix<int> mat1 = {
		{1,1,1,0},
		{0,3,1,2},
		{2,3,1,0},
		{1,0,2,1}
	};

	// ...or automatically using a function/lambda
	std::srand( std::time(nullptr) );
	Matrix<int> mat2(4, 3, []() { return std::rand() % 10; });

	// it is also possible to cast matrix types
	Matrix<double> mat3 = static_cast<Matrix<double>>(mat2);



	//  ------ PRINTING ------
	// You can print the Matrices with ostream
	std::cout << "mat1 : " << std::endl << mat1 << std::endl;
	std::cout << "mat2 : " << std::endl << mat2 << std::endl;
	std::cout << "mat3 : " << std::endl << mat2 << std::endl;



	//  ------ ACCESSORS ------
	// You can get the dimensions of the Matrix with getNrows/getMcols
	int nrows = mat1.getNrows(); // 4
	int ncols = mat1.getMcols(); // 4

	// You can access any item / row (as a vector) with brackets operator
	// NB - remember indexing starts at 0
	std::vector<int> row = mat1[0];
	int item = mat1[2][1]; // 3

	// You can get the Matrix representation of a row/column using getRows/getCols 
	// NB - remember indexing starts at 0
	Matrix<int> col_3 = mat1.getCols(2);
	std::cout
		<< "Third column of mat1 : " << std::endl
		<< col_3 << std::endl;

	// You can also pass a vector of rows to the function
	// NB - remember indexing starts at 0
	Matrix<int> rows_24 = mat1.getRows({1, 3});
	std::cout
		<< "Rows 2 and 4 of mat1 : " << std::endl
		<< rows_24 << std::endl;



	//  ------ APPENDING ------
	// You can append columns/rows of items vertically or horizontally,
	// but also vectors or matrices if the dimensions are compatible
	
	mat3.appendVer(0, 3); // append 3 rows of zeros below mat3
	std::cout
		<< "Appended 3 rows of zeros below mat3 : " << std::endl
		<< mat3 << std::endl;

	mat2.appendHor(mat1); // append mat1 to the right of mat2 
	std::cout
		<< "Appended mat1 to the right of mat2 : " << std::endl
		<< mat2 << std::endl;

	try {
		mat3.appendVer(mat1); // throw std::logic_error because dimensions do not fit
	} catch(std::logic_error e) {
		std::cout
			<< "Appending matrices with wrong dimensions gives this error : "
			<< e.what()
			<< std::endl;
	}



	//  ------ DROPPING ------
	// You can drop columns/rows of items using dropRows/dropCols,
	// with both int or vector<int> as argument
	// NB - remember indexing starts at 0

	mat3.dropRows({1});
	std::cout << std::endl
		<< "Dropped 2d row of mat3" << std::endl
		<< mat3 << std::endl;

	mat2.dropCols({3, 4, 5, 6});
	std::cout
		<< "dropped mat1 from mat2 (4 columns to the right)" << std::endl
		<< mat2 << std::endl;



	//  ------ METHODS ------
	//  there is some linear algebra methods
	//  (most can throw logic_error, see Matrix.h)

	try {

		// find the determinant
		int determinant = mat1.det();
		std::cout
			<< "determinant of mat1 : "
			<< determinant << std::endl;

		// find the inverse matrix (the Matrix returned is of the same type,
		// casting can be desired);
		Matrix<double> inverse = static_cast<Matrix<double>>(mat1).inv();
		std::cout << std::endl
			<< "inverse of mat1 : " << std::endl
			<< inverse << std::endl;

		// get the identity matrix
		Matrix<int> identity = static_cast<Matrix<double>>(mat1) * inverse;
		std::cout
			<< "mat1 * mat1.inverse() gives identity matrix : " << std::endl
			<< identity << std::endl;

		// transpose matrix
		Matrix<int> transposed_mat1 = mat1.transpose();
		std::cout
			<< "transposed mat1: " << std::endl
			<< transposed_mat1 << std::endl;

	} catch(std::logic_error e) {
		std::cout <<  e.what() << std::endl;
	}



	// ------ OPERATORS ------
	// Operators work as expected - see Matrix.h
	// ( can throw std::logic_error if there is dimension incompatibility)

	Matrix<int> mat4 = mat1 * (mat1 + mat1 - 1);
	
	return 0;
}
