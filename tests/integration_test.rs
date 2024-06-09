use sparse_matrix_lib::Matrix;

#[test]
fn test_create_matrix() {
    // Create a new sparse matrix with 3 rows and 4 columns
    let matrix = Matrix::create_matrix(3, 4);

    // Verify that the number of rows and columns is set correctly
    assert_eq!(matrix.get_rows(), 3);
    assert_eq!(matrix.get_cols(), 4);
}

#[test]
fn test_add_new_element() {
    // Create a new sparse matrix with 3 rows and 4 columns
    let mut matrix = Matrix::create_matrix(3, 4);

    // add item
    matrix.add_element(2, 2, 3.0);

    // assert value was pushed to vector
    assert_eq!(matrix.get_element(2, 2), 3.0);
}

#[test]
fn test_add_existing_element() {
    // Create a new sparse matrix with 3 rows and 4 columns
    let mut matrix = Matrix::create_matrix(3, 4);

    // add item
    matrix.add_element(2, 2, 3.0);
    matrix.add_element(2, 2, 5.0);

    // assert value changed
    assert_eq!(matrix.get_element(2, 2), 5.0);
}



#[test]
fn test_add_matrix() {
    
    // Create two sparse matrices
    let mut matrix1 = Matrix::create_matrix(2, 2);
    matrix1.add_element(0, 0, 1.0);
    matrix1.add_element(0, 1, 2.0);
    matrix1.add_element(1, 0, 3.0);
    matrix1.add_element(1, 1, 4.0);

    let mut matrix2 = Matrix::create_matrix(2, 2);
    matrix2.add_element(0, 0, 5.0);
    matrix2.add_element(0, 1, 6.0);
    matrix2.add_element(1, 0, 7.0);
    matrix2.add_element(1, 1, 8.0);

    // Add the matrices
    let result_matrix = Matrix::add(&matrix1,&matrix2);

    // Verify the result matrix
    assert_eq!(result_matrix.get_element(0, 0), 6.0);
    assert_eq!(result_matrix.get_element(0, 1), 8.0);
    assert_eq!(result_matrix.get_element(1, 0), 10.0);
    assert_eq!(result_matrix.get_element(1, 1), 12.0);
}

#[test]
fn test_subtract_matrix() {
    
    // Create two sparse matrices
    let mut matrix1 = Matrix::create_matrix(2, 2);
    matrix1.add_element(0, 0, 10.0);
    matrix1.add_element(0, 1, 2.0);
    matrix1.add_element(1, 0, 20.0);
    matrix1.add_element(1, 1, 8.0);

    let mut matrix2 = Matrix::create_matrix(2, 2);
    matrix2.add_element(0, 0, 5.0);
    matrix2.add_element(0, 1, 6.0);
    matrix2.add_element(1, 0, 7.0);
    matrix2.add_element(1, 1, 8.0);

    // Add the matrices
    let result_matrix = Matrix::subtract(&matrix1,&matrix2);

    // Verify the result matrix
    assert_eq!(result_matrix.get_element(0, 0), 5.0);
    assert_eq!(result_matrix.get_element(0, 1), -4.0);
    assert_eq!(result_matrix.get_element(1, 0), 13.0);
    assert_eq!(result_matrix.get_element(1, 1), 0.0);
}

#[test]
fn test_add_matrix_op() {
    
    // Create two sparse matrices
    let mut matrix1 = Matrix::create_matrix(2, 2);
    matrix1.add_element(0, 0, 1.0);
    matrix1.add_element(0, 1, 2.0);
    matrix1.add_element(1, 0, 3.0);
    matrix1.add_element(1, 1, 4.0);

    let mut matrix2 = Matrix::create_matrix(2, 2);
    matrix2.add_element(0, 0, 5.0);
    matrix2.add_element(0, 1, 6.0);
    matrix2.add_element(1, 0, 7.0);
    matrix2.add_element(1, 1, 8.0);

    // Add the matrices
    let result_matrix = matrix1 + matrix2;

    // Verify the result matrix
    assert_eq!(result_matrix.get_element(0, 0), 6.0);
    assert_eq!(result_matrix.get_element(0, 1), 8.0);
    assert_eq!(result_matrix.get_element(1, 0), 10.0);
    assert_eq!(result_matrix.get_element(1, 1), 12.0);
}

#[test]
fn test_subtract_matrix_op() {
    
    // Create two sparse matrices
    let mut matrix1 = Matrix::create_matrix(2, 2);
    matrix1.add_element(0, 0, 10.0);
    matrix1.add_element(0, 1, 2.0);
    matrix1.add_element(1, 0, 20.0);
    matrix1.add_element(1, 1, 8.0);

    let mut matrix2 = Matrix::create_matrix(2, 2);
    matrix2.add_element(0, 0, 5.0);
    matrix2.add_element(0, 1, 6.0);
    matrix2.add_element(1, 0, 7.0);
    matrix2.add_element(1, 1, 8.0);

    // Add the matrices
    let result_matrix = matrix1 - matrix2;

    // Verify the result matrix
    assert_eq!(result_matrix.get_element(0, 0), 5.0);
    assert_eq!(result_matrix.get_element(0, 1), -4.0);
    assert_eq!(result_matrix.get_element(1, 0), 13.0);
    assert_eq!(result_matrix.get_element(1, 1), 0.0);
}

#[test]
    fn test_multiply() {
        // Create two input matrices
        let mut matrix1 = Matrix::create_matrix(2, 3);
        matrix1.add_element(0, 0, 1.0);
        matrix1.add_element(0, 1, 2.0);
        matrix1.add_element(0, 2, 3.0);
        matrix1.add_element(1, 0, 4.0);
        matrix1.add_element(1, 1, 5.0);
        matrix1.add_element(1, 2, 6.0);

        let mut matrix2 = Matrix::create_matrix(3, 2);
        matrix2.add_element(0, 0, 7.0);
        matrix2.add_element(0, 1, 8.0);
        matrix2.add_element(1, 0, 9.0);
        matrix2.add_element(1, 1, 10.0);
        matrix2.add_element(2, 0, 11.0);
        matrix2.add_element(2, 1, 12.0);

        // Multiply the matrices
        let result = Matrix::multiply(&matrix1,&matrix2);

        // Assert the result matches the expected result
    
        assert_eq!(result.get_element(0, 0), 58.0);
        assert_eq!(result.get_element(0, 1), 64.0);
        assert_eq!(result.get_element(1, 0), 139.0);
        assert_eq!(result.get_element(1, 1), 154.0);
    }

#[test]
fn test_multiply_op() {
    // Create two input matrices
    let mut matrix1 = Matrix::create_matrix(2, 3);
    matrix1.add_element(0, 0, 1.0);
    matrix1.add_element(0, 1, 2.0);
    matrix1.add_element(0, 2, 3.0);
    matrix1.add_element(1, 0, 4.0);
    matrix1.add_element(1, 1, 5.0);
    matrix1.add_element(1, 2, 6.0);

    let mut matrix2 = Matrix::create_matrix(3, 2);
    matrix2.add_element(0, 0, 7.0);
    matrix2.add_element(0, 1, 8.0);
    matrix2.add_element(1, 0, 9.0);
    matrix2.add_element(1, 1, 10.0);
    matrix2.add_element(2, 0, 11.0);
    matrix2.add_element(2, 1, 12.0);

    // Multiply the matrices
    let result = matrix1 * matrix2;

    // Assert the result matches the expected result

    assert_eq!(result.get_element(0, 0), 58.0);
    assert_eq!(result.get_element(0, 1), 64.0);
    assert_eq!(result.get_element(1, 0), 139.0);
    assert_eq!(result.get_element(1, 1), 154.0);
}

#[test]
fn test_power() {
    // Create an input matrix
    let mut matrix = Matrix::create_matrix(2, 2);
    matrix.add_element(0, 0, 2.0);
    matrix.add_element(0, 1, 1.0);
    matrix.add_element(1, 0, 1.0);
    matrix.add_element(1, 1, 2.0);

    // Compute the cube of the matrix
    let result = Matrix::power(&matrix, 3);

    // Expected result matrix (matrix ^ 3)
    let mut expected_result = Matrix::create_matrix(2, 2);
    expected_result.add_element(0, 0, 14.0);
    expected_result.add_element(0, 1, 13.0);
    expected_result.add_element(1, 0, 13.0);
    expected_result.add_element(1, 1, 14.0);

    // Assert the result matches the expected result
    assert_eq!(*result.get_values(), *expected_result.get_values());
}

#[test]
fn test_power_op() {
    // Create an input matrix
    let mut matrix = Matrix::create_matrix(2, 2);
    matrix.add_element(0, 0, 2.0);
    matrix.add_element(0, 1, 1.0);
    matrix.add_element(1, 0, 1.0);
    matrix.add_element(1, 1, 2.0);

    // Compute the cube of the matrix
    let result = matrix ^ 3;

    // Expected result matrix (matrix ^ 3)
    let mut expected_result = Matrix::create_matrix(2, 2);
    expected_result.add_element(0, 0, 14.0);
    expected_result.add_element(0, 1, 13.0);
    expected_result.add_element(1, 0, 13.0);
    expected_result.add_element(1, 1, 14.0);

    // Assert the result matches the expected result
    assert_eq!(*result.get_values(), *expected_result.get_values());
}

#[test]
fn test_two_by_two_determinant_base() {
    // Create a 2x2 matrix
    let mut matrix = Matrix::create_matrix(2, 2);
    matrix.add_element(0, 0, 2.0);
    matrix.add_element(0, 1, 3.0);
    matrix.add_element(1, 0, 4.0);
    matrix.add_element(1, 1, 5.0);

    // Calculate the determinant
    let determinant = matrix.two_by_two_determinant();

    // Assert the determinant matches the expected value
    assert_eq!(determinant, -2.0);
}

#[test]
fn test_two_by_two_determinant_other_indices() {
    // Create a 2x2 matrix
    let mut matrix = Matrix::create_matrix(2, 2);
    matrix.add_element(1, 1, 2.0);
    matrix.add_element(1, 2, 3.0);
    matrix.add_element(2, 1, 4.0);
    matrix.add_element(2, 2, 5.0);

    // Calculate the determinant
    let determinant = matrix.two_by_two_determinant();

    // Assert the determinant matches the expected value
    assert_eq!(determinant, -2.0);
}

#[test]
fn test_submatrix() {
    // Create a 3x3 matrix
    let mut matrix = Matrix::create_matrix(3, 3);
    matrix.add_element(0, 0, 1.0);
    matrix.add_element(0, 1, 2.0);
    matrix.add_element(0, 2, 3.0);
    matrix.add_element(1, 0, 4.0);
    matrix.add_element(1, 1, 5.0);
    matrix.add_element(1, 2, 6.0);
    matrix.add_element(2, 0, 7.0);
    matrix.add_element(2, 1, 8.0);
    matrix.add_element(2, 2, 9.0);

    // Get the submatrix from row 1 to row 2 and column 1 to column 2
    let submatrix = matrix.submatrix(0, 0);

    // Create the expected submatrix
    let mut expected_submatrix = Matrix::create_matrix(2, 2);
    expected_submatrix.add_element(0, 0, 5.0);
    expected_submatrix.add_element(0, 1, 6.0);
    expected_submatrix.add_element(1, 0, 8.0);
    expected_submatrix.add_element(1, 1, 9.0);

    // Assert the submatrix matches the expected submatrix
    assert_eq!(*submatrix.get_values(), *expected_submatrix.get_values());
}

#[test]
fn test_determinant() {
    // Create a 4x4 matrix
    let mut matrix = Matrix::create_matrix(4, 4);
    matrix.add_element(0, 0, 1.0);
    matrix.add_element(0, 1, 2.0);
    matrix.add_element(0, 2, 3.0);
    matrix.add_element(0, 3, 4.0);
    matrix.add_element(1, 0, 5.0);
    matrix.add_element(1, 1, 6.0);
    matrix.add_element(1, 2, 7.0);
    matrix.add_element(1, 3, 8.0);
    matrix.add_element(2, 0, 9.0);
    matrix.add_element(2, 1, 10.0);
    matrix.add_element(2, 2, 11.0);
    matrix.add_element(2, 3, 12.0);
    matrix.add_element(3, 0, 13.0);
    matrix.add_element(3, 1, 14.0);
    matrix.add_element(3, 2, 15.0);
    matrix.add_element(3, 3, 16.0);

    // Calculate the determinant
    let determinant = matrix.determinant();
    // Assert the determinant matches the expected value
    assert_eq!(determinant, 0.0);
}

#[test]
fn test_lu_decomposition() {
    // Create a 3x3 matrix
    let mut matrix = Matrix::create_matrix(3, 3);
    matrix.add_element(0, 0, 1.0);
    matrix.add_element(0, 1, 2.0);
    matrix.add_element(0, 2, 3.0);
    matrix.add_element(1, 0, 4.0);
    matrix.add_element(1, 1, 5.0);
    matrix.add_element(1, 2, 6.0);
    matrix.add_element(2, 0, 7.0);
    matrix.add_element(2, 1, 8.0);
    matrix.add_element(2, 2, 9.0);

    // Perform LU decomposition
    let (lower, upper) = matrix.lu_decomposition();

    let out = lower * upper;
    
    for i in 0..matrix.get_rows() {
        for j in 0..matrix.get_cols() {
            assert_eq!(matrix.get_element(i, j), out.get_element(i, j));
        }
    }
}