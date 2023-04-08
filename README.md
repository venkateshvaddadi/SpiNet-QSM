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
The proposed SpiNet-QSM is a model-based deep learning technique for solving the QSM problem. It has unrolled iterative architecture.
SpiNet-QSM has two parts:
The data consistency term
The regularization term

  
# SpiNet-QSM equations
The proposed SpiNet-QSM solves the following optimization problem.
<img src="images/required_equations/relation between local filed and qsm.PNG" alt="spinet-QSM architecture" width=100% height=100%>


# SpiNet-QSM Architecture
<img src="images/SpiNet_QSM_architecture.png" alt="spinet-QSM architecture" width=100% height=100%>

</body>
</html>
