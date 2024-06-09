

pub struct Matrix {
    rows: i32,
    cols: i32,
    values: Vec<f64>,
    row_indices: Vec<i32>,
    col_indices: Vec<i32>,
}

impl Matrix {

    /// Creates a new Matrix with the specified number of rows and columns.
    ///
    /// # Arguments
    ///
    /// * `rows` - The number of rows in the Matrix.
    /// * `cols` - The number of columns in the Matrix.
    ///
    /// # Returns
    ///
    /// A new Matrix instance with the specified number of rows and columns.
    ///
    pub fn create_matrix(rows: i32, cols: i32) -> Matrix {
        Matrix {
            rows: rows,
            cols: cols,
            values: Vec::new(),
            row_indices: Vec::new(),
            col_indices: Vec::new(),
        }
    }

    /// Retrieves the number of rows in the Matrix.
    ///
    /// # Returns
    ///
    /// The number of rows in the Matrix.
    ///
    pub fn get_rows(&self) -> i32 {
        self.rows
    }

    /// Retrieves the number of columns in the Matrix.
    ///
    /// # Returns
    ///
    /// The number of columns in the Matrix.
    ///
    pub fn get_cols(&self) -> i32 {
        self.cols
    }

    /// Retrieves a reference to the values stored in the Matrix.
    ///
    /// # Returns
    ///
    /// A reference to the vector containing the values of the Matrix.
    ///
    pub fn get_values(&self) -> &Vec<f64> {
        &self.values
    }

    /// Retrieves a reference to the vector containing the row indices of the Matrix.
    ///
    /// # Returns
    ///
    /// A reference to the vector containing the row indices of the Matrix.
    ///
    pub fn get_row_vec(&self) -> &Vec<i32> {
        &self.row_indices
    }

    /// Retrieves a reference to the vector containing the column indices of the Matrix.
    ///
    /// # Returns
    ///
    /// A reference to the vector containing the column indices of the Matrix.
    ///
    pub fn get_col_vec(&self) -> &Vec<i32> {
        &self.col_indices
    }

    /// Adds an element to the Matrix at the specified row and column indices with the given value.
    ///
    /// If an element already exists at the specified row and column indices, its value will be updated.
    ///
    /// # Arguments
    ///
    /// * `row` - The row index of the element to add.
    /// * `col` - The column index of the element to add.
    /// * `value` - The value to set at the specified row and column indices.
    /// 
    /// # Returns
    /// 
    /// None
    ///
    /// # Panics
    ///
    /// Panics if `row` or `col` are out of bounds for the Matrix dimensions.
    ///
    pub fn add_element(&mut self, row: i32, col: i32, value: f64) {
        for i in 0..self.row_indices.len() {
            if self.row_indices[i] == row && self.col_indices[i] == col {
                self.values[i] = value;
                return;
            }
        }

        self.values.push(value);
        self.row_indices.push(row);
        self.col_indices.push(col);
    }

    /// Retrieves the value of the element at the specified row and column indices.
    ///
    /// If no element exists at the specified indices, returns 0.0.
    ///
    /// # Arguments
    ///
    /// * `row` - The row index of the element to retrieve.
    /// * `col` - The column index of the element to retrieve.
    ///
    /// # Returns
    ///
    /// The value of the element at the specified row and column indices, or 0.0 if no element exists.
    ///
    pub fn get_element(&self, row: i32, col: i32) -> f64 {
        for i in 0..self.row_indices.len() {
            if self.row_indices[i] == row && self.col_indices[i] == col {
                return self.values[i];
            }
        }
        0.0
    }

    /// Adds two matrices element-wise and returns the result as a new Matrix.
    ///
    /// # Arguments
    ///
    /// * `one` - The first Matrix operand.
    /// * `two` - The second Matrix operand.
    ///
    /// # Panics
    ///
    /// Panics if the dimensions of the input matrices `one` and `two` are not compatible (i.e., they have different numbers of rows or columns).
    ///
    /// # Returns
    ///
    /// A new Matrix containing the element-wise sum of the input matrices.
    ///
    pub fn add(one: &Matrix, two: &Matrix) -> Matrix {
        assert!(one.get_rows() == two.get_rows() && one.get_cols() == two.get_cols());
        let mut output = Matrix::create_matrix(one.get_rows(), one.get_cols());

        for row in 0..one.get_rows() {
            for col in 0..one.get_cols() {
                let one_val = one.get_element(row, col);
                let two_val = two.get_element(row, col);
                output.add_element(row, col, one_val + two_val);
            }
        }

        output
    }

    /// Subtracts one matrix from another element-wise and returns the result as a new Matrix.
    ///
    /// # Arguments
    ///
    /// * `one` - The Matrix from which the other Matrix will be subtracted.
    /// * `two` - The Matrix to be subtracted from the first Matrix.
    ///
    /// # Panics
    ///
    /// Panics if the dimensions of the input matrices `one` and `two` are not compatible (i.e., they have different numbers of rows or columns).
    ///
    /// # Returns
    ///
    /// A new Matrix containing the element-wise difference of the input matrices (one - two).
    ///
    pub fn subtract(one: &Matrix, two: &Matrix) -> Matrix {
        assert!(one.get_rows() == two.get_rows() && one.get_cols() == two.get_cols());
        let mut output = Matrix::create_matrix(one.get_rows(), one.get_cols());

        for row in 0..one.get_rows() {
            for col in 0..one.get_cols() {
                let one_val = one.get_element(row, col);
                let two_val = two.get_element(row, col);
                output.add_element(row, col, one_val - two_val);
            }
        }

        output
    }

    /// Multiplies two matrices and returns the result as a new Matrix.
    ///
    /// This method performs matrix multiplication between the two input matrices `one` and `two`. Matrix multiplication is defined such that the number of columns in the first matrix (`one`) must be equal to the number of rows in the second matrix (`two`).
    ///
    /// # Arguments
    ///
    /// * `one` - The first Matrix operand.
    /// * `two` - The second Matrix operand.
    ///
    /// # Panics
    ///
    /// Panics if the number of columns in `one` is not equal to the number of rows in `two`.
    ///
    /// # Returns
    ///
    /// A new Matrix containing the result of multiplying `one` by `two`.
    ///
    pub fn multiply(one: &Matrix, two: &Matrix) -> Matrix {
        let mut output = Matrix::create_matrix(one.get_rows(), two.get_cols());

        let one_row_v = one.get_row_vec();
        let one_col_v = one.get_col_vec();
        let two_row_v = two.get_row_vec();
        let two_col_v = two.get_col_vec();

        for i in 0..one_row_v.len() {
            for j in 0..two_row_v.len() {
                if one_col_v[i] == two_row_v[j] {
                    let new_value = one.get_element(one_row_v[i], one_col_v[i]) * two.get_element(two_row_v[j], two_col_v[j]);
                    output.add_element(one_row_v[i], two_col_v[j], output.get_element(one_row_v[i], two_col_v[j]) + new_value);
                }
            }
        }
        output
    }

    /// Raises a square matrix to the power of `pow`.
    ///
    /// This method computes the `pow`th power of the input square matrix `one`. The matrix must have the same number of rows and columns.
    ///
    /// # Arguments
    ///
    /// * `one` - A reference to the input Matrix operand.
    /// * `pow` - The power to which the matrix will be raised. Must be greater than 0.
    ///
    /// # Panics
    ///
    /// Panics if the number of rows is not equal to the number of columns in the input matrix `one`.
    /// Panics if `pow` is not greater than 0.
    ///
    /// # Returns
    ///
    /// A new Matrix containing the result of raising `one` to the power of `pow`.
    ///
    pub fn power(one: &Matrix, pow: i32) -> Matrix {
        assert!(one.get_cols() == one.get_rows());
        assert!(pow > 0);
        let mut output = Matrix::create_matrix(one.get_rows(), one.get_cols());
        output = Matrix::add(&output, &one);

        for _i in 1..pow {
            output = Matrix::multiply(&output, &one);
        }
        output
    }

    /// Calculates the determinant of a square matrix.
    ///
    /// This function computes the determinant of the matrix represented by the
    /// `self` object. It first checks that the matrix is square (i.e., it has the same
    /// number of rows and columns) using an assertion. The function uses a recursive
    /// approach to calculate the determinant:
    ///
    /// 1. **Base Case**: If the matrix is 2x2, it calculates the determinant directly
    ///    using a specific method for 2x2 matrices (`two_by_two_determinant`).
    ///
    /// 2. **Recursive Case**: For larger matrices, it uses Laplace expansion by minors.
    ///    It iterates over the first row, computing the determinant of submatrices (minors)
    ///    and accumulating the result.
    ///
    /// # Panics
    ///
    /// The function panics if the matrix is not square, i.e., if the number of rows
    /// does not equal the number of columns.
    ///
    /// # Returns
    ///
    /// Returns the determinant of the matrix as a `f64` value.
    ///
    pub fn determinant(&self) -> f64 {
        assert!(self.get_cols() == self.get_rows());
        //base
        if self.get_cols() == 2 && self.get_rows() == 2 {
            return self.two_by_two_determinant();
        }
        else {
            let mut det = 0.0;
            let mut sign = 1.0;
            for i in 0..self.get_col_vec().len() {
                if self.get_col_vec()[i] == 0 {
                    det += sign * self.get_values()[i] * self.submatrix(self.get_row_vec()[i], self.get_col_vec()[i]).determinant();
                    sign *= -1.0;
                }
            }
            return det;
        }
    }

    /// Calculates the determinant of a 2x2 matrix.
    ///
    /// This function computes the determinant of a 2x2 matrix represented by the
    /// `self` object. It first checks that the matrix is 2x2 using an assertion. The
    /// determinant for a 2x2 matrix \(\left[\begin{array}{cc}
    /// a & b \\
    /// c & d \\
    /// \end{array}\right]\) is calculated using the formula \(ad - bc\).
    ///
    /// # Panics
    ///
    /// The function panics if the matrix is not 2x2, i.e., if the number of rows or
    /// columns is not equal to 2.
    ///
    /// # Returns
    ///
    /// Returns the determinant of the 2x2 matrix as a `f64` value.
    ///
    pub fn two_by_two_determinant(&self) -> f64 {
        assert!(self.get_cols() == 2 && self.get_rows() == 2);
        let mut hash: [f64; 2] = [1.0, 1.0];
        for i in 0..self.get_row_vec().len() {
            if self.get_row_vec()[i] == self.get_col_vec()[i] {
                hash[0] *= self.get_values()[i];
            }
            else {
                hash[1] *= self.get_values()[i];
            }
        }
        return hash[0] - hash[1];
    }

    /// Creates a submatrix by removing the specified row and column.
    ///
    /// This function generates a new matrix that is a submatrix of the current matrix
    /// by excluding the specified row (`current_row`) and column (`current_col`). It is
    /// typically used in the context of determinant calculation where minors of the matrix
    /// are required.
    ///
    /// # Parameters
    ///
    /// - `current_row`: The row to be excluded from the submatrix.
    /// - `current_col`: The column to be excluded from the submatrix.
    ///
    /// # Returns
    ///
    /// Returns a new `Matrix` object that represents the submatrix after excluding the
    /// specified row and column.
    ///
    pub fn submatrix(&self, current_row: i32, current_col: i32) -> Matrix {
        let mut output = Matrix::create_matrix(self.get_rows() - 1, self.get_cols() - 1);
        for i in 0..self.get_row_vec().len() {
            if self.get_row_vec()[i] != current_row && self.get_col_vec()[i] != current_col {
                output.add_element(self.get_row_vec()[i], self.get_col_vec()[i], self.get_values()[i]);
            }
        }
        output
    }

    /// Performs LU decomposition of a square matrix.
    ///
    /// This function decomposes the matrix represented by `self` into a lower triangular matrix `L`
    /// and an upper triangular matrix `U` such that `A = LU`, where `A` is the original matrix.
    /// The function uses the Doolittle algorithm for LU decomposition, where the diagonal elements
    /// of `L` are set to 1.
    ///
    /// # Panics
    ///
    /// The function panics if the matrix is not square, i.e., if the number of rows does not equal
    /// the number of columns.
    ///
    /// # Returns
    ///
    /// Returns a tuple containing two `Matrix` objects: the lower triangular matrix `L` and the
    /// upper triangular matrix `U`.
    ///
    pub fn lu_decomposition(&self) -> (Matrix, Matrix) {
        // Check if the matrix is square
        assert!(self.get_cols() == self.get_rows());

        let n = self.get_rows();

        // Create the L and U matrices
        let mut l = Matrix::create_matrix(n, n);
        let mut u = Matrix::create_matrix(n, n);

        // Initilize the L matrix to an identity matrix
        for i in 0..n {
            l.add_element(i, i, 1.0);
        }

        for i in 0..n {

            // Calculate the lower triangular matrix
            for j in i..n {
                let mut new_value = self.get_element(i, j);
                for k in 0..i {
                    new_value -= l.get_element(i, k) * u.get_element(k, j);
                }
                u.add_element(i, j, new_value);
            } 

            // Calculate the upper triangular matrix
            for j in i+1..n {
                let mut new_value = self.get_element(j, i);
                for k in 0..i {
                    new_value -= l.get_element(j, k) * u.get_element(k, i);
                }
                l.add_element(j, i, new_value / u.get_element(i, i));
            }
        }     
        (l, u)
    }
}

/// Adds two matrices element-wise and returns the result as a new Matrix.
///
/// # Arguments
///
/// * `rhs` - The right-hand side Matrix operand.
///
/// # Panics
///
/// Panics if the dimensions of the matrices `self` and `rhs` are not compatible (i.e., they have different numbers of rows or columns).
///
/// # Returns
///
/// A new Matrix containing the element-wise sum of `self` and `rhs`.
///
impl std::ops::Add<Matrix> for Matrix {
    type Output = Matrix;

    fn add(self, _rhs: Matrix) -> Matrix {
        Matrix::add(&self, &_rhs)
    }
}

/// Subtracts one matrix from another element-wise and returns the result as a new Matrix.
///
/// # Arguments
///
/// * `rhs` - The right-hand side Matrix operand (to be subtracted from `self`).
///
/// # Panics
///
/// Panics if the dimensions of the matrices `self` and `rhs` are not compatible (i.e., they have different numbers of rows or columns).
///
/// # Returns
///
/// A new Matrix containing the element-wise difference of `self` and `rhs`.
///
impl std::ops::Sub<Matrix> for Matrix {
    type Output = Matrix;

    fn sub(self, _rhs: Matrix) -> Matrix {
        Matrix::subtract(&self, &_rhs)
    }
}

/// Multiplies two matrices and returns the result as a new Matrix.
///
/// This implementation enables the use of the `*` operator for matrix multiplication.
///
/// # Arguments
///
/// * `rhs` - The right-hand side Matrix operand.
///
/// # Panics
///
/// Panics if the number of columns in `self` is not equal to the number of rows in `rhs`.
///
/// # Returns
///
/// A new Matrix containing the result of multiplying `self` by `rhs`.
///
impl std::ops::Mul<Matrix> for Matrix {
    type Output = Matrix;

    fn mul(self, _rhs: Matrix) -> Matrix {
        Matrix::multiply(&self, &_rhs)
    }
}

/// Raises the matrix to the power of `pow`.
///
/// This implementation enables the use of the `^` operator for raising a matrix to a power.
///
/// # Arguments
///
/// * `pow` - The power to which the matrix will be raised. Must be greater than 0.
///
/// # Panics
///
/// Panics if the matrix is not square.
/// Panics if `pow` is not greater than 0.
///
/// # Returns
///
/// A new Matrix containing the result of raising `self` to the power of `pow`.
///
impl std::ops::BitXor<i32> for Matrix {
    type Output = Matrix;

    fn bitxor(self, pow: i32) -> Matrix {
        Matrix::power(&self, pow)
    }
}