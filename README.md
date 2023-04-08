<html>
<head>
</head>
<body>

<left>
<h1>SpiNet QSM: Model-based Deep Learning with Schatten p‐norm regularization for QSM reconstruction </h1>
</left>
<h2> QSM problem</h2>
QSM provides information about the underlying magnetic susceptibility distribution of a sample from MRI phase measurements. It is useful in the clinical diagnosis of diseases  like Demyelination, Calcification, and Parkinson’s disease.
  
The mathematical relation for the QSM reconstruction is:
<center>
<img src="images/required_equations/qsm_problem_relation.png" alt="spinet-QSM architecture" width=75% height=75%>
</center>
For solving QSM problem, it is required peform dipole deconvolution with local field. It is very cruical step in the QSM solving. Unfortunately, this it is an illposed problem.
<img src="images/required_equations/relation between local filed and qsm.PNG" alt="spinet-QSM architecture" width=100% height=100%>

# SpiNet-QSM 
The proposed SpiNet-QSM is a model-based deep learning technique for solving the QSM problem. The proposed approach can enforce p-norm (0 < p ≤ 2) on trainable regularizer, where its norm parameter (p) is trainable (automatically chosen).

SpiNet-QSM has two parts: data consistency term and regularization term.

  
# SpiNet-QSM equations
The proposed SpiNet-QSM solves the following optimization problem.
<img src="images/required_equations/SpiNet_QSM_optimization_equation.png" alt="spinet-QSM architecture" width=100% height=100%>

In this equation, the norm parameter (**p**) of the regularization term and regularization parameter(**𝜆**) are learnable for the QSM problem. 

Here, J(χ) has been solved in iteratively using the majorization-minimization approach.

# SpiNet-QSM Architecture
<img src="images/SpiNet_QSM_architecture.png" alt="spinet-QSM architecture" width=100% height=100%>

# SpiNet-QSM as a unrolled architecture
<img src="images/unrolled_iterative_architecture.png" alt="spinet-QSM architecture" width=100% height=100%>

</body>
</html>
