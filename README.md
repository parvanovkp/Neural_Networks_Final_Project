# Neural Networks Final Project

This is my final project repository for CSCI 5922-873: Neural Nets and Deep Learning (Fall 2023). The code you will find here was developed by me for the purpose of demonstrating the usefullness of Physics-Informed Neural Networks for solving ordinary differential equations. More specifically the project focusses on solving the unforced damped pendulum problem given as

\begin{equation}
\frac{d^2\theta}{dt^2} + b\frac{d\theta}{dt} + \frac{g}{L}\sin(\theta) = 0
\end{equation}

where:
\begin{itemize}
    \item \(\theta\) is the angular displacement.
    \item \(b\) is the damping coefficient.
    \item \(g\) is the acceleration due to gravity.
    \item \(L\) is the length of the pendulum.
\end{itemize}
