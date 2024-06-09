# Sparcify
## Overview:
This project was inspired by two classes that I took. The first was Linear Algebra, where I learned about various matrix operations that I have implemented in this project, as well as the NumPy library, 
which served as an inspiration for this project. The second was ECE 220, where I encountered sparse matrices. One of our assignments in that class was to implement addition and multiplication for a sparse matrix.

I started this project for two main reasons: first, because I thought it would be interesting to connect the concepts from both classes, and second, because I wanted to experiment with Rust.

Below I go through some of the operations that I personally thought were interesting.

## Storage
The storage of sparse matrices in memory distinguishes this library from others. 
It stores the coordinates of a non-zero value and the value itself as an unordered list in memory. 
This approach avoids wasting space on storing zeros, a major benefit of sparse matrices.

## Multiplication:
### Initial Implementation
My first implementation of matrix multiplication was the naive algorithm. 
Where for each element in the output matrix you used the equation below. 
Where $C$ is the output matrix and $A$ and $B$ are the matrices that are being multiplied. 
$i$ and $j$ are the row and column of the element respectively and $n$ is the number of columns in A minus one. 

$$C_{ij}=A_{i0}B_{0j} + A_{i1}B_{1j} + ...A_{in}B_{nj}  $$

### With the Code:

```
pub fn multiply(one: &Matrix, two: &Matrix) -> Matrix {
    assert!(one.get_cols() == two.get_rows(), "Matrix dimensions must agree for multiplication");

    let rows = one.get_rows();
    let cols = two.get_cols();
    let mut output = Matrix::create_matrix(rows, cols);

    for row in 0..rows {
        for col in 0..cols {
            let mut sum = 0.0;
            for n in 0..one.get_cols() {
                sum += one.get_element(row, n) * two.get_element(n, col);
            }
            if sum != 0.0 {
                output.add_element(row, col, sum);
            }
        }
    }

    output
}
```
### Improved Implementation
The previous implementation is inefficient as it multiplies and adds a lot of zeros. 
This problem only grows as we multiply matrices that are more and more sparse. 
My solution to this started with thinking about how each non-zero element in a matrix contributes to the output matrix. 
A non-zero element in the matrix A lets say $A_{01}$ will have a contribution to all elements of the output in the $0th$ row(from the equation above). 
The contribution is $C_{0j}=A_{01}B_{1j}$ where $B_{1j}$ is each element in the $1th$ row of $B$

### With the Code:

```
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
```
## LU Decomposition
### Implementation:
The LU Decomposition funciton implements the Dolittle algorithm. Which is:

For each row $i$ from 0 to $n-1$ where n is the number of rows in the square matrix:

  For each column $j$ from $i$ to $n-1$:
  
$$U_{ij} = A_{ij} - \sum_{k=0}^{i-1}L_{ik}U_{ki}$$

  For each column $j$ from $i+1$ to $n-1$:

$$L_{ij} = \frac{1}{U_{ii}}(A_{ji} - \sum_{k=0}^{i-1}L_{jk}U_{ki})$$

Where $A$ is the original matrix, $L$ is a lower triangular matrix, and $U$ is an upper triangular matrix such that $A=LU$

## Determinant
The determinant function currently calculates the determinant manually, recursively breaking down the matrix until reaching a $2 \times 2$ matrix. 
In the future, I plan to implement this function using LU decomposition.

## Future
This project is a work in progress. In the future, I intend to add more functionality such as inverse matrices and QR decomposition.
