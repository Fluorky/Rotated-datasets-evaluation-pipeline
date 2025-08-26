---
header-includes:
  - \usepackage{graphicx}
  - \usepackage{multicol}
  - \usepackage{ragged2e}
  - \usepackage{tocloft}
  - \renewcommand{\cftsecleader}{\cftdotfill{\cftdotsep}}
---


\begin{titlepage}
\centering


\includegraphics[width=0.7 \textwidth]{media/ul_logo.png}

\vspace{1cm}

{\LARGE \textbf{Maciej Bujalski}} \\[1cm]

\RaggedRight
{\large
\textbf{Field of study:} Computer Science\\
\textbf{Specialization:} Applied Computer Science\\
\textbf{Focus:} Mobile Applications\\
\textbf{Student ID:} 386012\\
}

\vspace{2.5cm}

\centering
{\Large \textbf{Rotation-Invariant Neural Networks}} \\[2cm]

\begin{flushright}
\large
\textbf{Master's Thesis} \\
supervised by \\
Dr. Krzysztof Podlaski \\
Department of Intelligent Systems, \\
Faculty of Physics and Applied Computer Science, University of Łódź
\end{flushright}

\vfill

{\large Łódź 2025}

\end{titlepage}

\newpage

\tableofcontents

\newpage

# Introduction

Images surround us everywhere: from smartphone photos, satellite imagery,
and city surveillance, through product catalogs and quality control on
production lines, to driver-assistance systems. Although modern image
recognition models perform remarkably well, in practice they can be
sensitive to seemingly minor changes such as rotating an object by a few
degrees or slightly tilting the camera. What is natural and immediately
recognizable to a human (a road sign at an angle, a digit rotated on
paper) can be problematic for a classical convolutional neural network.
The core difficulty is the lack of natural invariance to rotations:
standard CNNs are “by definition” better at handling translations than
rotations [@goodfellow2016deep; @dumoulin2016guide].

In recent years, several approaches have been proposed to close this
gap. One is augmenting the data with rotated examples – which improves
robustness but increases training time and does not guarantee
generalization to all angles. Another approach is designing architectures
with built-in geometry: group-equivariant networks (G-CNN,
E(2)-equivariant) [@cohen2016group; @kim2020cycnn], cyclic networks
(CyCNN; in particular **CyVGG** and **CyResNet**) operating across
multiple orientations, and transformations into polar coordinates
(linear-polar and log-polar), which “straighten” rotations into
translations. The common goal is for the model to recognize “the same”
object regardless of orientation, without aggressive data duplication.

This thesis focuses on the practical evaluation of these approaches.
Datasets were prepared including handwritten digits, road signs (in color
and grayscale), and synthetic 3D objects projected onto 2D (LEGO bricks),
which were then extended with controlled rotations. Selected
rotation-invariant architectures and their baseline variants were
implemented and compared in **PyTorch** [@paszke2019pytorch], measuring
the impact of transformations (linear-polar vs. log-polar), architectural
choices, and angular ranges on prediction quality. The experiments were
carried out on **NVIDIA GeForce RTX 3070 Ti 8 GB** and **RTX 3060 12 GB**
GPUs, which significantly reduced training times and enabled a broad
experimental survey; the runtime environment was standardized using
**Docker** for reproducibility.

The aim of this work is not only to show that it is possible to achieve
rotation invariance, but more importantly to indicate **when** and **at
what cost** it can be achieved, which techniques yield the greatest
improvement over classical CNNs, how they affect stability and training
speed, and which configurations are the most practical in real-world
applications (OCR, traffic sign recognition, analysis of technical
objects). The following chapters present the theoretical background,
datasets and augmentation strategies, architectures, experimental
environment, evaluation protocols, results, and conclusions.
